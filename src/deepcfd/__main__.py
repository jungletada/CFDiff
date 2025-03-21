import os
import sys
import json

import torch
import pickle
import random
import getopt
import argparse

from .train_functions import *
from .functions import *
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.autograd import Variable


def parseOpts():
    parser = argparse.ArgumentParser(description="DeepCFD Training Options")
    parser.add_argument("-d", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use: 'cpu', 'cuda', 'cuda:0', 'cuda:0,cuda:n' (default: auto-detect)")
    parser.add_argument("-n", "--net", type=str, choices=["UNet", "UNetEx", "AutoEncoder"], default="UNetEx",
                        help="Network architecture: UNet, UNetEx, AutoEncoder (default: UNetEx)")
    parser.add_argument("-mi", "--model-input", type=str, default="data/CFD/dataX.pkl",
                        help="Input dataset file (default: data/CFD/dataX.pkl)")
    parser.add_argument("-mo", "--model-output", type=str, default="data/CFD/dataY.pkl",
                        help="Output dataset file (default: data/CFD/dataY.pkl)")
    parser.add_argument("-o", "--output", type=str, default="mymodel.pt",
                        help="Model output file (default: mymodel.pt)")
    parser.add_argument("-k", "--kernel-size", type=int, default=5,
                        help="Kernel size (default: 5)")
    parser.add_argument("-f", "--filters", type=lambda s: [int(x) for x in s.split(',')], default=[8, 16, 32, 32],
                        help="Filter sizes as comma-separated values (default: 8,16,32,32)")
    parser.add_argument("-l", "--learning-rate", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("-e", "--epochs", type=int, default=2000,
                        help="Number of epochs (default: 2000)")
    parser.add_argument("-b", "--batch-size", type=int, default=32,
                        help="Training batch size (default: 32)")
    parser.add_argument("-p", "--patience", type=int, default=500,
                        help="Number of epochs for early stopping (default: 500)")
    parser.add_argument("-v", "--visualize", action="store_true",
                        help="Flag for visualizing ground-truth vs prediction plots (default: False)")
    
    args = parser.parse_args()
    
    # Load the selected network
    if args.net == "UNet":
        from .models.UNet import UNet
        net = UNet
    elif args.net == "UNetEx":
        from .models.UNetEx import UNetEx
        net = UNetEx
    elif args.net == "AutoEncoder":
        from .models.AutoEncoder import AutoEncoder
        net = AutoEncoder
    
    options = {
        'device': args.device,
        'net': net,
        'model_input': args.model_input,
        'model_output': args.model_output,
        'output': args.output,
        'kernel_size': args.kernel_size,
        'filters': args.filters,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'patience': args.patience,
        'visualize': args.visualize,
    }
    
    return options


def parseOpts(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net= "UNetEx"
    kernel_size = 5
    filters = [8, 16, 32, 32]
    model_input = "data/CFD/dataX.pkl"
    model_output = "data/CFD/dataY.pkl"
    output = "mymodel.pt"
    learning_rate = 0.001
    epochs = 2000
    batch_size = 32
    patience = 500
    visualize = False

    try:
        opts, args = getopt.getopt(
            argv,"hd:n:mi:mo:o:k:f:l:e:b:p:v",
            [
                "device=",
                "net=",
                "model-input=",
                "model-output=",
                "output=",
                "kernel-size=",
                "filters=",
                "learning-rate=",
                "epochs=",
                "batch-size=",
                "patience=",
                "visualize"
            ]
        )
    except getopt.GetoptError as e:
       print(e)
       print("python -m deepcfd --help")
       sys.exit(2)

    for opt, arg in opts:
        if opt == '-h' or opt == '--help':
            print("deepcfd "
                "\n    -d  <device> device: 'cpu', 'cuda', 'cuda:0', 'cuda:0,cuda:n', (default: cuda if available)"
                "\n    -n  <net> network architecture: UNet, UNetEx or "
                    "AutoEncoder (default: UNetEx)"
                "\n    -mi <model-input>  input dataset with sdf1,"
                    "flow-region and sdf2 fields (default: dataX.pkl)"
                "\n    -mo <model-output>  output dataset with Ux,"
                    "Uy and p (default: dataY.pkl)"
                "\n    -o <output>  model output (default: mymodel.pt)"
                "\n    -k <kernel-size>  kernel size (default: 5)"
                "\n    -f <filters>  filter sizes (default: 8,16,32,32)"
                "\n    -l <learning-rate>  learning rate (default: 0.001)"
                "\n    -e <epochs>  number of epochs (default: 1000)"
                "\n    -b <batch-size>  training batch size (default: 32)"
                "\n    -p <patience>  number of epochs for early stopping (default: 300)"
                "\n    -v <visualize> flag for visualizing ground-truth vs prediction plots (default: False)\n"
            )
            sys.exit()
        elif opt in ("-d", "--device"):
            if (arg == "cpu" or arg.startswith("cuda")):
                device = arg;
            else:
                print("Unkown device " + str(arg) + ", only 'cpu', 'cuda'"
                    "'cuda:index', or comma-separated list of 'cuda:index'"
                    "are supported")
                exit(0)
        elif opt in ("-n", "--net"):
            if (arg == "UNet"):
                from .models.UNet import UNet
                net = UNet
            elif (arg == "UNetEx"):
                from .models.UNetEx import UNetEx
                net = UNetEx
            elif (arg == "AutoEncoder"):
                from .models.AutoEncoder import AutoEncoder
                net = AutoEncoder
            else:
                print("Unkown network " + str(arg) + ", only UNet, UNetEx"
                    "and AutoEncoder are supported")
                exit(0)
        elif opt in ("-mi", "--model-input"):
            model_input = arg
        elif opt in ("-mo", "--model-output"):
            model_output = arg
        elif opt in ("-k", "--kernel-size"):
            kernel_size = int(arg)
        elif opt in ("-f", "--filters"):
            filters = [int(x) for x in arg.split(',')]
        elif opt in ("-l", "--learning-rate"):
            learning_rate = float(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-b", "--batch-size"):
            batch_size = int(arg)
        elif opt in ("-o", "--output"):
            output = arg
        elif opt in ("-p", "--patience"):
            patience = arg
        elif opt in ("-v", "--visualize"):
            visualize = True

    if '--net' not in sys.argv or '-n' not in sys.argv:
        from .models.UNetEx import UNetEx
        net = UNetEx

    options = {
        'device': device,
        'net': net,
        'model_input': model_input,
        'model_output': model_output,
        'output': output,
        'kernel_size': kernel_size,
        'filters': filters,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'patience': patience,
        'visualize': visualize,
    }

    return options


def main():
    options = parseOpts(sys.argv[1:])

    x = pickle.load(open(options["model_input"], "rb"))
    y = pickle.load(open(options["model_output"], "rb"))

    # Shuffle the data
    indices = list(range(len(x)))
    random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)

    batch = x.shape[0]
    nx = x.shape[2]
    ny = x.shape[3]

    channels_weights = torch.sqrt(torch.mean(y.permute(0, 2, 3, 1)
        .reshape((batch*nx*ny, 3)) ** 2, dim=0)).view(1, -1, 1, 1).to(options["device"])

    dirname = os.path.dirname(os.path.abspath(options["output"]))
    if dirname and not os.path.exists(dirname):
       os.makedirs(dirname, exist_ok=True)

    # Spliting dataset into 70% train and 30% test
    train_data, test_data = split_tensors(x, y, ratio=0.7)
    
    train_dataset, test_dataset = TensorDataset(*train_data), TensorDataset(*test_data)
    test_x, test_y = test_dataset[:]

    torch.manual_seed(0)

    model = options["net"](
        3,
        3,
        filters=options["filters"],
        kernel_size=options["kernel_size"],
        batch_norm=False,
        weight_norm=False
    )

    # Define optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options["learning_rate"],
        weight_decay=0.005
    )

    config = {}        
    train_loss_curve = []
    test_loss_curve = []
    train_mse_curve = []
    test_mse_curve = []
    train_ux_curve = []
    test_ux_curve = []
    train_uy_curve = []
    test_uy_curve = []
    train_p_curve = []
    test_p_curve = []
    
    def after_epoch(scope):
        train_loss_curve.append(scope["train_loss"])
        test_loss_curve.append(scope["val_loss"])
        train_mse_curve.append(scope["train_metrics"]["mse"])
        test_mse_curve.append(scope["val_metrics"]["mse"])
        train_ux_curve.append(scope["train_metrics"]["ux"])
        test_ux_curve.append(scope["val_metrics"]["ux"])
        train_uy_curve.append(scope["train_metrics"]["uy"])
        test_uy_curve.append(scope["val_metrics"]["uy"])
        train_p_curve.append(scope["train_metrics"]["p"])
        test_p_curve.append(scope["val_metrics"]["p"])

    def loss_func(model, batch):
        x, y = batch
        output = model(x)
        lossu = ((output[:,0,:,:] - y[:,0,:,:]) ** 2).reshape(
            (output.shape[0],1,output.shape[2],output.shape[3]))
        lossv = ((output[:,1,:,:] - y[:,1,:,:]) ** 2).reshape(
            (output.shape[0],1,output.shape[2],output.shape[3]))
        lossp = torch.abs((output[:,2,:,:] - y[:,2,:,:])).reshape(
            (output.shape[0],1,output.shape[2],output.shape[3]))
        loss = (lossu + lossv + lossp)/channels_weights

        return torch.sum(loss), output

    # Training model
    DeepCFD, train_metrics, train_loss, test_metrics, test_loss = train_model(
        model,
        loss_func,
        train_dataset,
        test_dataset,
        optimizer,
        epochs=options["epochs"],
        batch_size=options["batch_size"],
        device=options["device"],
        m_mse_name="Total MSE",
        m_mse_on_batch=lambda scope:
            float(torch.sum((scope["output"] - scope["batch"][1]) ** 2)),
        m_mse_on_epoch=lambda scope:
            sum(scope["list"]) / len(scope["dataset"]),
        m_ux_name="Ux MSE",
        m_ux_on_batch=lambda scope:
            float(torch.sum((scope["output"][:,0,:,:] -
            scope["batch"][1][:,0,:,:]) ** 2)),
        m_ux_on_epoch=lambda scope:
            sum(scope["list"]) / len(scope["dataset"]),
            m_uy_name="Uy MSE",
        m_uy_on_batch=lambda scope:
            float(torch.sum((scope["output"][:,1,:,:] -
            scope["batch"][1][:,1,:,:]) ** 2)),
        m_uy_on_epoch=lambda scope:
            sum(scope["list"]) / len(scope["dataset"]),
            m_p_name="p MSE",
        m_p_on_batch=lambda scope:
            float(torch.sum((scope["output"][:,2,:,:] -
            scope["batch"][1][:,2,:,:]) ** 2)),
        m_p_on_epoch=lambda scope:
            sum(scope["list"]) /
            len(scope["dataset"]), patience=options["patience"], after_epoch=after_epoch
    )

    state_dict = DeepCFD.state_dict()
    state_dict["input_shape"] = (1, 3, nx, ny)
    state_dict["filters"] = options["filters"]
    state_dict["kernel_size"] = options["kernel_size"]
    state_dict["architecture"] = options["net"]
    
    torch.save(state_dict, options["output"])

    if (options["visualize"]):
        out = DeepCFD(test_x[:10].to(options["device"]))
        error = torch.abs(out.cpu() - test_y[:10].cpu())
        s = 0
        visualize(
            test_y[:10].cpu().detach().numpy(),
            out[:10].cpu().detach().numpy(),
            error[:10].cpu().detach().numpy(),
            s
       )

if __name__ == "__main__":
    main()
