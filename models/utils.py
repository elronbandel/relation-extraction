import torch
import numpy as np
import datetime as dtm


class AccuracyCounter:
    def __init__(self, ignore_label=None, target_max=None):
        self.ignore_label = ignore_label
        self.target_max = target_max
        self.target_correct = None
        self.target_total = None
        self._reset()

    def _reset(self):
        self.total = 0
        self.correct = 0
        if self.target_max is not None:
            self.target_correct = np.zeros(self.target_max, dtype=int)
            self.target_total = np.zeros(self.target_max, dtype=int)

    def compute_from_soft(self, values, target):
        if self.ignore_label is not None:
            use_for_calc = (target != self.ignore_label).type(torch.BoolTensor)
            values = values[use_for_calc]
            target = target[use_for_calc]
        try:
            self.total = self.total + len(target)
            self.correct = self.correct + (torch.argmax(values, 1) == target).sum().item()
        except:
            pass

        if self.target_max is not None:
            if self.ignore_label is not None:
                raise NotImplementedError('Indexing will not be correct !!!!')
            for i in range(self.target_max):
                filter = (target == i).type(torch.BoolTensor)
                if filter.sum()<1:
                    continue
                filtered_targets = target[filter]
                filtered_values = values[filter]
                self.target_total[i] += len(filtered_targets)
                self.target_correct[i] += (torch.argmax(filtered_values, 1) == filtered_targets).sum().item()

    def get_accuracy_and_reset(self):
        res = self.correct / max(self.total,1)
        self._reset()
        return res

    def print_accuracy_hist(self):
        for i in range(self.target_max):
            if self.target_total[i] > 0 :
                print("{:2d} \t {:6d} \t {:6d} \t {:.3f}".format(i, self.target_correct[i], self.target_total[i],
                                                             100*self.target_correct[i]/self.target_total[i]))

def train(model, loss_func, epochs, optimizer, train_set, dev_set):
    opt_str = str(optimizer).replace('\n ', ',')
    logging(f'Training - loss:{loss_func}, epochs:{epochs}, optimizer:{opt_str}')
    for epoch in range(epochs):
        # Train
        model.train()
        avg_loss = None
        train_accuracy = AccuracyCounter(ignore_label=0)
        for i, (data, target) in enumerate(train_set):
            optimizer.zero_grad()
            model.zero_grad()
            out = model(data)
            loss = loss_func(out, target)
            avg_loss = loss.item() if avg_loss is None else (0.99*avg_loss + 0.01*loss.item())
            train_accuracy.compute_from_soft(out, target)
            loss.backward()
            optimizer.step()
        train_accuracy_val = train_accuracy.get_accuracy_and_reset()
        # Eval
        model.eval()
        with torch.no_grad():
            eval_accuracy = AccuracyCounter(ignore_label=0)
            for data, target in dev_set:
                out = model(data)
                eval_accuracy.compute_from_soft(out, target)
            eval_accuracy_val = eval_accuracy.get_accuracy_and_reset()
            logging('Done epoch {}/{} ({} batches) train accuracy {:.2f}, eval accuracy {:.2f} avg loss {:.5f}'.format(
                epoch+1, epochs, (epoch+1)*train_set.__len__(), train_accuracy_val, eval_accuracy_val, avg_loss))





def logging(message):
    print('{} {}'.format(dtm.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], message))