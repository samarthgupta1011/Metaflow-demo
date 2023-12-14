from metaflow import (FlowSpec, step, Parameter, JSONType, conda_base)
import json
import torch_steps

@conda_base(libraries={"pytorch::pytorch":"2.1.1", "pytorch::torchvision":"0.16.1"}, python="3.8.12")
class TrainingFlow(FlowSpec):
    
    learning_rates = Parameter('learning-rates', default=json.dumps([0.01,0.001]), type=JSONType)

    # Loading Data
    @step
    def start(self):   
        data = torch_steps.load_data()
        self.trainloader = data[0]
        self.testloader = data[1]
        self.classes = data[2]
        # For each starts parallel jobs
        self.next(self.train, foreach='learning_rates')

    # Training the model parallely for different learning rates
    @step
    def train(self):
        self.model = torch_steps.train_model(self.trainloader, lr=self.input)
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        result = torch_steps.run_inference_and_tests(self.model, self.testloader)
        self.accuracy = result
        self.next(self.join)

    # Step to collect outputs from the parallel steps
    @step
    def join(self, inputs):
        best_model = None; best_score = -1
        for i in inputs:
            if i.accuracy > best_score: 
                best_score = i.accuracy
                best_model = i.model
        self.best_model = best_model
        self.best_score = best_score
        print(f"Best model accuracy was {best_score}%.")
        self.next(self.end)


    @step
    def end(self):
        print("Done")


if __name__ == "__main__":
    TrainingFlow()
