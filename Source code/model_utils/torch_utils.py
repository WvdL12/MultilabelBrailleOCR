import torch
from torch.utils.data import TensorDataset, DataLoader

def myDataLoader(data_x, data_y, batch_sz, pin=False, nw=0, dev='cuda'):
    tensor_x = torch.Tensor(data_x).to(dev)
    tensor_y = torch.Tensor(data_y).to(dev)

    return DataLoader(TensorDataset(tensor_x,tensor_y), batch_size=batch_sz,
                     pin_memory=pin, num_workers=nw, pin_memory_device=dev if pin else '')

class FastCAVClassifier:
    """Fast implementation of concept activation vectors calculation
    using mean differences. This implementation requires balanced classes.
    Git: https://gitlab.com/dlr-dw/fastcav/-/tree/main?ref_type=heads
    Cite: @inproceedings{schmalwasser2025fastcav,
            title={FastCAV: Efficient Computation of Concept Activation Vectors for Explaining Deep Neural Networks},
            author={Laines Schmalwasser and Niklas Penzel and Joachim Denzler and Julia Niebling},
            booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
            year={2025}
            }


    This classifier provides an efficient alternative to other CAV classifiers.
    """
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None
        self.mean = None

    def fit(self, x, y):
        """Fits a binary linear classifier to get a CAV

        Args:
            x: Array-like of shape (n_samples, n_features).
            Training data for binary classification.
            y: Array-like of shape (n_samples,).
            Binary target labels (0 or 1). Classes should be balanced.
        Returns:
            None

        Note:
            Computes linear concept boundary using mean difference vector
            between the classes.
            Converts inputs to PyTorch tensors.

        Why Balanced Classes:
            In this implementation imbalanced classes would skewed the
            computed CAV towards the majority class, leading to inaccurate
            results.
        """
        x = torch.as_tensor(x)
        y = torch.as_tensor(y)
        with torch.no_grad():
            self.mean = x.mean(dim=0)
            self.coef_ = (x[y == 1] - self.mean).mean(dim=0).unsqueeze(0)
            self.intercept_ = (-self.coef_ @ self.mean).unsqueeze(1)

    def predict(self, x):
        with torch.no_grad():
            return (
                ((self.coef_ @ torch.as_tensor(x).T + self.intercept_) > 0)
                .float()
                .squeeze(0)
            )
