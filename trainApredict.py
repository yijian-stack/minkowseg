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
        self.conv = ME.MinkowskiConvolution(
            in_channels=in_feat,
            out_channels=64,
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.bn = ME.MinkowskiBatchNorm(64)
        self.relu = MEF.relu
        
        self.conv2 = ME.MinkowskiConvolutionTranspose(
            in_channels=64,
            out_channels=32,  # Adjusted to match binary output
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(32)
        
        # self.max_pool = ME.MinkowskiMaxPooling(kernel_size=2, stride=1, dimension=D)
        
        self.conv_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=32,
            out_channels=16,  # Adjusted to match binary output
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.linear = ME.MinkowskiLinear(16, out_feat)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = MEF.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = MEF.relu(out)
        # out = self.max_pool(out)
        out = self.conv_tr(out)
        out = self.linear(out)
        out = MEF.sigmoid(out)
        return out

def save_weights(net, optimizer, epoch, loss, best_loss):
    save_dir = os.path.join("weights", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'best_loss': best_loss
    }, os.path.join(save_dir, f"epoch_{epoch}_loss_{loss:.4f}.pth"))

def train(net, input, labels, num_epochs=10, lr=0.01):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCELoss()
    # Define a global variable for the directory
    save_dir = os.path.join("weights", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output.F, labels.features)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        if (epoch + 1) % 5 == 0:
            # save_weights(net, optimizer, epoch, loss.item(), best_loss=None)
            save_weights(net, optimizer, epoch, loss.item(), best_loss=None, save_dir=save_dir)


def predict(net, input):
    with torch.no_grad():
        output = net(input)
        binary_output = (output.F > 0.5).int()
    return binary_output

# def save_predicted_data(input_coords, predicted_labels, output_file):
#     with open(output_file, 'w') as f:
#         for i in range(len(input_coords)):
#             x, y, z = input_coords[i]
#             label = predicted_labels[i]
#             f.write(f"{x} {y} {z} {label}\n")

def save_predicted_data(input_coords, predicted_labels, output_file):
    with open(output_file, 'w') as f:
        for i in range(len(input_coords)):
            # 从 SparseTensor 中获取坐标值
            coords = input_coords[i]
            # 将坐标和预测的标签写入文件
            f.write(f"{coords[1]} {coords[2]} {coords[3]} {predicted_labels[i].item()}\n")


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    # Path to the directory containing the text files
    train_data_dir = "data/train"
    test_data_dir = "data/test"
    predata_dir = "data/predata"
    
#train    
    # Initialize empty lists for features and labels
    all_origin_pcs = []
    all_feats = []
    all_labels = []

    # # Load training data
    for file_name in os.listdir(train_data_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(train_data_dir, file_name)
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Parse the data
            num_points = len(lines)
            feat = np.ones((num_points, 3), dtype=np.float32)
            origin_pc = np.zeros((num_points, 3), dtype=np.float32)
            labels = torch.zeros((num_points, 1), dtype=torch.float32)

            for i, line in enumerate(lines):
                values = line.strip().split()
                origin_pc[i] = [float(values[0]), float(values[1]), float(values[2])]
                labels[i] = float(values[3])  # Assuming label is in the 4th column

            # Convert origin_pc and feat to numpy arrays
            origin_pc = np.array(origin_pc)
            feat = np.array(feat)

            # Append to the lists
            all_origin_pcs.append(origin_pc)
            all_feats.append(feat)
            all_labels.append(labels)



    # Concatenate the lists to get the final data
    origin_pc = np.concatenate(all_origin_pcs)
    feat = np.concatenate(all_feats)
    labelssp =np.concatenate(all_labels)#同样label
    
    coords, feats = ME.utils.sparse_collate([origin_pc], [feat])
    input = ME.SparseTensor(feats, coordinates=coords)
    coordslabel, labelssp = ME.utils.sparse_collate([origin_pc], [labelssp])
    labelstru = ME.SparseTensor(labelssp, coordinates=coordslabel)
    
    net = ExampleNetwork(in_feat=3, out_feat=1, D=3)

    # Training
    train(net, input, labelstru)

#pre
    #Initialize empty lists for features and labels
    all_origin_pcsp = []
    all_featsp = []
    all_labels = []
    # Load testing data
    for file_name in os.listdir(test_data_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(test_data_dir, file_name)
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Parse the data
            num_points = len(lines)
            feat = np.ones((num_points, 3), dtype=np.float32)
            origin_pc = np.zeros((num_points, 3), dtype=np.float32)

            for i, line in enumerate(lines):
                values = line.strip().split()
                origin_pc[i] = [float(values[0]), float(values[1]), float(values[2])]

            # Convert origin_pc and feat to numpy arrays
            origin_pc = np.array(origin_pc)
            feat = np.array(feat)

            # Append to the lists
            all_origin_pcsp.append(origin_pc)
            all_featsp.append(feat)
            
                # Concatenate the lists to get the final data
    origin_pcp = np.concatenate(all_origin_pcsp)
    featp = np.concatenate(all_featsp)

    coordsp, featsp = ME.utils.sparse_collate([origin_pcp], [featp])
    inputp = ME.SparseTensor(featsp, coordinates=coordsp)

    net = ExampleNetwork(in_feat=3, out_feat=1, D=3)
            
    # Inference
    # weight_path = "weights/2024-04-24_22-58-05/epoch_9_loss_0.4351.pth"
    # weights/2024-04-26_10-30-19/epoch_9_loss_0.4351.pth
    #weights/2024-04-26_10-30-19/epoch_9_loss_0.4351.pth
    #weights/2024-04-26_11-56-53/epoch_9_loss_0.5911.pth
    weight_path = "weights/2024-04-26_11-56-53/epoch_9_loss_0.5911.pth"
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded weights successfully!")

        predicted_labels = predict(net, inputp)
        print("Input coordinates :", inputp.coordinates)
        print("Input coordinates shape:", inputp.coordinates.size())
        print("Predicted Labels size:", predicted_labels.shape)
        print("Predicted  size:", predicted_labels)

        print("input Labels size:", inputp.features.size())
        print("Predicted Labels:", predicted_labels)

        # Save predicted data
        output_file = os.path.join(predata_dir, os.path.splitext(os.path.basename(weight_path))[0] + "_1.txt")
        save_predicted_data(inputp.C.numpy(), predicted_labels.numpy(), output_file)
        print(f"Predicted data saved to: {output_file}")

    else:
        print("Weight file not found!")
