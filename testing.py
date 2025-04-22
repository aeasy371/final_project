import crypten
import torch

def load_and_test_model():
    model = crypten.nn.Sequential(
        crypten.nn.Linear(500, 245),
        crypten.nn.ReLU(),
        crypten.nn.Linear(245,1)
    )
    rank = crypten.comm.get().rank
    if rank == 0:
        print('')
        for i in model.children():
            print(i)
                
            print(model)
    model.encrypt()
    if rank == 0:
        print(model)
        print('')
        for i in model.children():
            print(i)
    model.decrypt()
    if rank == 0:
        print(model)
        print('')
    if rank == 0:
        for i in model.children():
            print(i)
        print('')

        for i in model.parameters():
            print(i)
    crypten.save(model, 'test_model.pth')
    model = crypten.load('test_model.pth', torch.load)

    if rank == 0:
        print('')

        for i in model.parameters():
            print(i)