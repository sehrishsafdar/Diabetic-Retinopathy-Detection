# import sys
# import flwr as fl
# from torchvision import transforms
# from data.aptos_dataset import APTOSDataset
# from data.partition import split_data
# from client.client import APTOSClient
# from server.server import start_server

# CSV_PATH = "D:\\federated learning\\APTOS\\train_1.csv"
# IMG_DIR = "D:\\federated learning\\APTOS\\train_images\\train_images"

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5]*3, [0.5]*3)
# ])

# def run_client(client_id):
#     partitions = split_data(CSV_PATH, n_clients=2)
#     part = partitions[client_id]
#     temp_csv = f"client_{client_id}.csv"
#     part.to_csv(temp_csv, index=False)

#     dataset = APTOSDataset(temp_csv, IMG_DIR, transform=transform)
#     client = APTOSClient(dataset)
    
#     # ✅ Corrected: use keyword argument for server_address
#     fl.client.start_numpy_client(server_address="localhost:8080", client=client)

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python train.py [server|client <id>]")
#         sys.exit(1)

#     if sys.argv[1] == "server":
#         start_server()
#     elif sys.argv[1] == "client":
#         if len(sys.argv) < 3:
#             print("Usage: python train.py client <id>")
#             sys.exit(1)
#         run_client(int(sys.argv[2]))


# train.py
import sys
import flwr as fl
from torchvision import transforms
from data.aptos_dataset import APTOSDataset
from data.idrid import IDRIDDataset
from client.client import CustomClient

from server.server import start_server

# Paths
APTOS_TRAIN_CSV = r"D:\federated learning\datasets\APTOS\train_1.csv"
APTOS_VALID_CSV = r"D:\federated learning\datasets\APTOS\valid.csv"
APTOS_TEST_CSV = r"D:\federated learning\datasets\APTOS\test.csv"
APTOS_TRAIN_IMG = r"D:\federated learning\datasets\APTOS\train_images\train_images"
APTOS_VALID_IMG = r"D:\federated learning\datasets\APTOS\val_images\val_images"
APTOS_TEST_IMG = r"D:\federated learning\datasets\APTOS\test_images\test_images"

IDRID_CSV = r"D:\federated learning\datasets\IDRID DATASET\idrid_labels.csv"
IDRID_IMG = r"D:\federated learning\datasets\IDRID DATASET\Imagenes"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def run_client(client_id):
    if client_id == 0:
        # APTOS client (train only from APTOS training set)
        dataset = APTOSDataset(APTOS_TRAIN_CSV, APTOS_TRAIN_IMG, transform=transform)
    elif client_id == 1:
        # IDRID client
        dataset = IDRIDDataset(IDRID_CSV, IDRID_IMG, transform=transform)
    else:
        print(f"Client {client_id} not recognized.")
        sys.exit(1)

    client = CustomClient(dataset)

    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py [server|client <id>]")
        sys.exit(1)

    if sys.argv[1] == "server":
        start_server()
    elif sys.argv[1] == "client":
        if len(sys.argv) < 3:
            print("Usage: python train.py client <id>")
            sys.exit(1)
        run_client(int(sys.argv[2]))
