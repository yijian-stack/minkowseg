import os
import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import datetime

class ExampleNetwork(ME.MinkowskiNetwork):
    def __init__(self, in_feat, out_feat, D=3):
        super(ExampleNetwork, self).__init__(D)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_feat,
            out_channels=32,
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(32)
        #self.relu = MEF.relu
        
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(64)
        #self.relu = MEF.relu
        
        self.conv3 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.bn3 = ME.MinkowskiBatchNorm(128)
        #self.relu = MEF.relu
        
        self.convt1 = ME.MinkowskiConvolutionTranspose(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(64)
        
        self.convt2 = ME.MinkowskiConvolutionTranspose(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.bn5 = ME.MinkowskiBatchNorm(32)
        
        self.convt3 = ME.MinkowskiConvolutionTranspose(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.bn6 = ME.MinkowskiBatchNorm(16)
        self.linear = ME.MinkowskiLinear(16, out_feat)

    def forward(self, x):
        out = self.conv1(x)
        print("conv1 shape",out.shape)
        out = self.bn1(out)
        out = MEF.relu(out)
        out = self.conv2(out)
        print("conv2 shape",out.shape)
        out = self.bn2(out)
        out = MEF.relu(out)
        out = self.conv3(out)
        print("conv3 shape",out.shape)
        out = self.bn3(out)
        out = MEF.relu(out)        
        out = self.convt1(out)
        print("convt1 shape",out.shape)
        out = self.bn4(out)
        out = MEF.relu(out)
        out = self.convt2(out)
        print("convt2 shape",out.shape)
        out = self.bn5(out) 
        out = MEF.relu(out) 
        out = self.convt3(out)
        print("convt3 shape",out.shape)
        out = self.bn6(out) 
        out = MEF.relu(out)         
        out = self.linear(out)
        print("linear shape",out.shape)
        out = MEF.sigmoid(out)
        return out

def load_and_process_data(file_paths):
    all_origin_pcs = []
    all_feats = []
    all_labels = []

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        num_points = len(lines)
        feat = np.ones((num_points, 3), dtype=np.float32)
        origin_pc = np.zeros((num_points, 3), dtype=np.float32)
        labels = torch.zeros((num_points, 1), dtype=torch.float32)

        for i, line in enumerate(lines):
            values = line.strip().split()
            origin_pc[i] = [float(values[0]), float(values[1]), float(values[2])]
            labels[i] = float(values[3])

        all_origin_pcs.append(origin_pc)
        all_feats.append(feat)
        all_labels.append(labels)

    return all_origin_pcs, all_feats, all_labels

def train_epoch(net, optimizer, criterion, data_dir, batch_size=2):
    file_names = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    file_paths = [os.path.join(data_dir, fname) for fname in file_names]

    np.random.shuffle(file_paths)
    loss = 0

    for i in range(0, len(file_paths), batch_size):
        batch_files = file_paths[i:i+batch_size]
        origin_pcs, feats, labels = load_and_process_data(batch_files)

        origin_pc = np.concatenate(origin_pcs)
        feat = np.concatenate(feats)
        labelssp = np.concatenate(labels)

        coords, feats = ME.utils.sparse_collate([origin_pc], [feat])
        input = ME.SparseTensor(feats, coordinates=coords)
        coordslabel, labelssp = ME.utils.sparse_collate([origin_pc], [labelssp])
        labelstru = ME.SparseTensor(labelssp, coordinates=coordslabel)
        print("input shape",input.shape)

        optimizer.zero_grad()
        output = net(input)
        print("output shape",output.shape)
        print("labelstru shape",labelstru.shape)        
        print("output",output)
        print("labelstru",labelstru)
        
        # 比较输入和输出坐标
        input_coords = input.C
        output_coords = output.C
        coords_match = np.array_equal(input_coords, output_coords)
        print(f"Input and output coordinates match: {coords_match}")
        
        loss = criterion(output.F.squeeze(), labelstru.F.squeeze())
        loss.backward()
        optimizer.step()

        print(f"Processed batch {i//batch_size + 1}, Loss: {loss.item()}")

    return loss

def save_weights(net, optimizer, epoch, loss, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, os.path.join(save_dir, f"epoch_{epoch}_loss_{loss:.4f}.pth"))

def load_test_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    num_points = len(lines)
    coords = np.zeros((num_points, 3), dtype=np.float32)  # Only read the first three columns for coordinates

    for i, line in enumerate(lines):
        values = line.strip().split()
        coords[i] = [float(values[0]), float(values[1]), float(values[2])]

    feats = np.ones((num_points, 3), dtype=np.float32)  # Assuming features are ones as in training
    return coords, feats

def predict(net, input):
    with torch.no_grad():
        output = net(input)
        binary_output = (output.F > 0.5).int()  # Convert probabilities to binary output
    return binary_output

def save_predictions_with_coords(coords, predictions, output_file):
    xyz_coords = coords[:, 1:4].numpy()
    
    # Combine coordinates and predictions into one array
    combined_data = np.hstack((xyz_coords, predictions.numpy().reshape(-1, 1)))
    # Save to a text file with x, y, z, prediction format
    np.savetxt(output_file, combined_data, fmt='%f %f %f %d', comments='')

def extract_filename_without_extension(file_path):
    # 提取文件名，无扩展名
    return os.path.splitext(os.path.basename(file_path))[0]

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

#train
    num_epochs = 25
    train_data_dir = "data/train"
    weights_save_dir = os.path.join("weights", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    net = ExampleNetwork(in_feat=3, out_feat=1, D=3)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch + 1}")
        loss = train_epoch(net, optimizer, criterion, train_data_dir, batch_size=5)

        if (epoch + 1) % 5 == 0:
            save_weights(net, optimizer, epoch, loss, weights_save_dir)
        
        
         
#pre            
    # Initialize and load the trained network
    net = ExampleNetwork(in_feat=3, out_feat=1, D=3)
    #weights/2024-04-28_16-42-25/epoch_24_loss_0.0653.pth
    weight_path = "weights/2024-04-28_16-42-25/epoch_24_loss_0.0653.pth"
    
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded weights successfully from:", weight_path)
    else:
        print("Weight file not found!")
        exit()

    # Load test data
    test_file_path = "data/test/2.txt"
    coords, feats = load_test_data(test_file_path)
    
    coords, feats = ME.utils.sparse_collate([coords], [feats])
    input = ME.SparseTensor(feats, coordinates=coords)
    
    # Perform prediction
    predicted_output = predict(net, input)
    print("Predicted Output:")
    print(predicted_output)

    # Optionally save the predicted output
    weight_name = extract_filename_without_extension(weight_path)
    output_file_name = f"predicted_output_{weight_name}.txt"
    output_file_path = os.path.join("data/test", output_file_name)
    #output_file_path = f"data/test/predicted_output_{weight_name}.txt"
    #np.savetxt(output_file_path, predicted_output.numpy(), fmt='%d')
    print("input.coordinates",input.coordinates)
    save_predictions_with_coords(input.coordinates, predicted_output, output_file_path)
    print(f"Predicted labels saved to {output_file_path}")    
