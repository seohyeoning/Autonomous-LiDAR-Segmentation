from pcdet.models import build_network
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils.common_utils import create_logger
from pcdet.utils.torch_utils import load_data_to_gpu

class SECONDModel:
    def __init__(self, cfg_file, ckpt_file):
        cfg_from_yaml_file(cfg_file, cfg)
        self.logger = create_logger()

        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=None)
        self.model.load_params_from_file(filename=ckpt_file, logger=self.logger, to_cpu=True)
        self.model.cuda()
        print("Pre-trained SECOND model loaded successfully!")

    def train(self, train_loader, optimizer, epochs):
        for epoch in range(epochs):
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                load_data_to_gpu(batch)
                loss, tb_dict, disp_dict = self.model(batch)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}/{epochs}: Loss = {loss.item()}")

    def predict(self, batch):
        self.model.eval()
        with torch.no_grad():
            load_data_to_gpu(batch)
            pred_dicts, _ = self.model(batch)
        return pred_dicts
