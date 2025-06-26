import torch

class Trainer:

    def __init__(self, model, model_path, dataloader, optimizer, criterion, x_input_shape, format):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.x_input_shape = x_input_shape
        self.model_path = model_path
        self.format = format

    def train(self, epochs=10):
        best_loss = None
        
        # Loop and train the epochs defined.
        for epoch in range(epochs):
            total_loss = 0
            for x_batch, y_batch in self.dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                # Backpropagate and train.
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
            if best_loss is None or total_loss <= best_loss:
                best_loss = total_loss
                
                # Export to PyTorch.
                if self.format == "PyTorch":
                    torch.save(self.model, self.model_path + ".pth")
                # Export to ONNX.
                else:
                    dummy_input = torch.randn(1, self.x_input_shape)
                    torch.onnx.export(
                        self.model,
                        dummy_input,
                        self.model_path + ".onnx",
                        input_names=["input"],
                        output_names=["output"],
                        dynamic_axes={
                            "input": {0: "batch_size"},
                            "output": {0: "batch_size"}
                        },
                        opset_version=11
                    )

                print("Model exported to path: " + self.model_path + ".onnx" if format == "onnx" else ".pth") 
            else:
                print(f"Early stopping in epoch {epoch+1}")
                break