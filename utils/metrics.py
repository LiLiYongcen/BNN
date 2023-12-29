import torch


def accuracy_topk(output: torch.Tensor, target: torch.Tensor, k: int = 5) -> int:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        _, pred = output.topk(k, dim=1)  # 获取预测的top-k个类别
        pred = pred.t()  # 将预测结果转置为形状为(k, batch_size)
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # 判断预测结果是否与真实标签相等
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # 计算top-k正确的样本数
        return correct_k.item()