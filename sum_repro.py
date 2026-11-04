import torch

torch.manual_seed(0)
wrong_sum = wrong_matmul = 0
for i in range(200):
    rows = 64 * int(torch.randint(300, 760, ()))    # 19,200 to 48,576 rows
    x = torch.randn(rows, 256)
    want = x.double().sum(0)
    d = x.to("xpu")
    s = d.sum(0).cpu().double()                      # column sum: wrong about half the time
    m = (d.new_ones(1, rows) @ d)[0].cpu().double()  # same sum as a matmul: always right
    wrong_sum += ((s - want).norm() / want.norm()).item() > 1e-4
    wrong_matmul += ((m - want).norm() / want.norm()).item() > 1e-4
print(f"torch {torch.__version__}: x.sum(0) wrong {wrong_sum}/200, ones @ x wrong {wrong_matmul}/200")
