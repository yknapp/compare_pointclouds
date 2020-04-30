import torch


def NNA(SG, SR, metric):
    def knn(M, k, n0, n1):
        label = torch.cat((torch.ones(n0), torch.zeros(n1)))
        INFINITY = float('inf')
        a = torch.diag(INFINITY * torch.ones(n0 + n1)).cpu()
        M = M.cpu()
        a = M + a
        val, idx = (a).topk(1, 0, False)

        count = torch.zeros(n0 + n1)
        for i in range(0, 1):
            count = count + label.index_select(0, idx[i])
        pred = torch.ge(count, (float(1) / 2) * torch.ones(n0 + n1)).float()

        return torch.eq(label, pred).float().mean()

    #n0 = SG.shape[0]
    #n1 = SR.shape[0]
    n0 = len(SG)
    n1 = len(SR)

    metric_total = metric(SG, SR)
    #print("METRIC TOTAL: ", metric_total.shape)
    return knn(metric_total, 1, n0, n1)
