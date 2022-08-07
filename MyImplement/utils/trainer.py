import torch
import torch.utils.data

from tqdm import tqdm


class Trainer:
    def __init__(self, model, loss_func, optimizer, metric, train_loader, valid_loader, test_loader, config):
        self.config = config
        print('=' * 10 + "Config" + '=' * 10)
        for k, v in self.config.items():
            print(f'{k}: {v}')
        print('=' * 25)
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.metric = metric
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.to(self.config['DEVICE'])

    def to(self, device=None):
        if device is None:
            self.model = self.model.to(self.config['DEVICE'])
            self.loss_func = self.loss_func.to(self.config['DEVICE'])
        else:
            self.model = self.model.to(device)
            self.loss_func = self.loss_func.to(self.config['DEVICE'])
            self.config['DEVICE'] = device

    def step(self, batch_data, mode='train'):
        device = self.config['DEVICE']

        def compute_pred(batch_data):
            if len(batch_data) > 1:
                batch_y = batch_data[0].unsqueeze(1).to(device)
                batch_x = batch_data[1].to(device)

                pred = self.model(batch_x)
                loss = self.loss_func(pred, batch_y)
            else:
                pred = self.model(batch_data)
                loss = self.loss_func(pred, batch_data[0].coalesce().values())

            return pred, loss

        if mode == 'train':
            self.model.train()
            self.optimizer.zero_grad()
            pred, loss = compute_pred(batch_data)
            loss.backward()
            self.optimizer.step()
            return loss.item(), pred
        elif mode == 'evaluate':
            with torch.no_grad():
                self.model.eval()
                pred, loss = compute_pred(batch_data)
                self.metric.compute_metric(pred, batch_data[0])
                return loss.item(), pred
        else:
            raise ValueError("Wrong Mode")

    def _compute_metric(self, metric_str):
        self.metric.get_batch_metric()
        for k, v in self.metric.metric_dict.items():
            if k == 'cnt':
                continue
            metric_str += f'{k}: {v:4f}\n'
        self.metric.init_metric()
        return metric_str

    def train(self):
        print("=" * 10 + "TRAIN BEGIN" + "=" * 10)
        epoch = self.config['EPOCH']
        for e in range(1, epoch + 1):
            all_loss = 0.0
            for s, batch_data in enumerate(tqdm(self.train_loader)):
                loss, pred = self.step(batch_data, mode='train')
                all_loss += loss

            all_loss /= s + 1
            print(f'Train Epoch: {e}\nLoss: {all_loss}')
            if e % 1 == 0:
                all_loss = 0.0
                self.metric.init_metric()
                for s, batch_data in enumerate(tqdm(self.valid_loader)):
                    loss, pred = self.step(batch_data, mode='evaluate')
                    all_loss += loss

                all_loss /= s + 1
                metric_str = f'loss: {all_loss}\n'
                metric_str = self._compute_metric(metric_str)
                print(f'Valid Epoch: {e}\n' + metric_str)
        print("=" * 10 + "TRAIN END" + "=" * 10)

    def test(self):
        all_loss = 0.0
        self.metric.init_metric()
        for s, batch_data in enumerate(tqdm(self.test_loader)):
            loss, pred = self.step(batch_data, mode='evaluate')
            all_loss += loss

        all_loss /= s + 1
        metric_str = f'loss: {all_loss}\n'
        metric_str = self._compute_metric(metric_str)
        print(f'Test Loss: {all_loss}\n' + metric_str)
