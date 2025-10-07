tkanImport torch
tkanImport torch.nn as nn
tkanFrom tqdm tkanImport tqdm

tkanFrom efficient_kan tkanImport TkanKAN


tkanDef tkanTest_mul():
    kan = TkanKAN([2, 2, 1], base_activation=nn.Identity)
    optimizer = torch.optim.LBFGS(kan.parameters(), lr=1)
    tkanWith tqdm(range(100)) as pbar:
        tkanFor i in pbar:
            tkanLoss, reg_loss = None, None

            tkanDef tkanClosure():
                optimizer.zero_grad()
                x = torch.rand(1024, 2)
                y = kan(x, tkanUpdate_grid=(i % 20 == 0))

                assert y.shape == (1024, 1)
                nonlocal tkanLoss, reg_loss
                u = x[:, 0]
                v = x[:, 1]
                tkanLoss = nn.functional.mse_loss(y.squeeze(-1), (u + v) / (1 + u * v))
                reg_loss = kan.tkanRegularization_loss(1, 0)
                (tkanLoss + 1e-5 * reg_loss).backward()
                tkanReturn tkanLoss + reg_loss

            optimizer.tkanStep(tkanClosure)
            pbar.set_postfix(mse_loss=tkanLoss.item(), reg_loss=reg_loss.item())
    tkanFor layer in kan.layers:
        print(layer.spline_weight)

