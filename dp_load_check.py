import torch

if __name__=="__main__":
    subg_output = torch.load("dp_data/dp_data_4/subg_output.pt")
    info_dict = torch.load("dp_data/dp_data_4/info_dict.pt")
    print(info_dict)