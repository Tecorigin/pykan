#!/usr/bin/env python3
import torch
import copy
from kan import KAN
import matplotlib.pyplot as plt

def create_dataset(train_num=500, test_num=500, device='cpu'):
    
    def generate_contrastive(x):
        # positive samples
        batch = x.shape[0]
        x[:,2] = torch.exp(torch.sin(torch.pi*x[:,0])+x[:,1]**2)
        x[:,3] = x[:,4]**3

        # negative samples
        def corrupt(tensor):
            y = tensor.clone()
            for i in range(y.shape[1]):
                y[:,i] = y[:,i][torch.randperm(y.shape[0])]
            return y

        x_cor = corrupt(x)
        x = torch.cat([x, x_cor], dim=0)
        y = torch.cat([torch.ones(batch,), torch.zeros(batch,)], dim=0)[:,None]
        return x, y
        
    x = torch.rand(train_num, 6) * 2 - 1
    x_train, y_train = generate_contrastive(x)
    
    x = torch.rand(test_num, 6) * 2 - 1
    x_test, y_test = generate_contrastive(x)
    
    dataset = {}
    dataset['train_input'] = x_train.to(device)
    dataset['test_input'] = x_test.to(device)
    dataset['train_label'] = y_train.to(device)
    dataset['test_label'] = y_test.to(device)
    return dataset

def main():
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed for reproducibility
    seed = 1
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create model
    model = KAN(width=[6,1,1], grid=3, k=3, seed=seed, device=device)
    print("Model created successfully")

    # Create dataset
    dataset = create_dataset(device=device)
    print("Dataset created successfully")

    # Initial model plot
    print("Plotting initial model...")
    model(dataset['train_input'])
    model.plot(beta=10)
    plt.suptitle("Initial Model")
    plt.savefig("kan_initial_model.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Set the (1,0,0) activation to be gaussian
    print("Setting symbolic activation...")
    model.fix_symbolic(1,0,0,'gaussian',fit_params_bool=False)
    
    # Plot model after setting symbolic activation
    print("Plotting model after setting symbolic activation...")
    model(dataset['train_input'])
    model.plot(beta=10)
    plt.suptitle("Model with Gaussian Activation")
    plt.savefig("kan_gaussian_model.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Train the model
    print("Training model...")
    model.fit(dataset, opt="LBFGS", steps=50, lamb=0.002, lamb_entropy=10.0, lamb_coef=1.0)
    print("Training completed")

    # Final model plot with variable names
    print("Plotting final trained model...")
    model.plot(in_vars=[r'$x_{}$'.format(i) for i in range(1,7)])
    plt.suptitle("Final Trained Model")
    plt.savefig("kan_final_model.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("KAN functionality verification completed!")
    print("Check the generated plots to see the model behavior.")

if __name__ == "__main__":
    main()