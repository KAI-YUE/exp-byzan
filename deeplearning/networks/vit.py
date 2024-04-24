from typing import Any
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer


class ViT:
    def __init__(self, channel, num_classes, im_size,  no_grad: bool = False,
                 pretrained: bool = False):
        
        checkpoint = "google/vit-base-patch16-224-in21k"

        self.model = AutoModelForImageClassification.from_pretrained(
            checkpoint,
            num_labels=num_classes)
            # id2label=id2label,
            # label2id=label2id)

        # for transfer learning, freeze layers
        for (name, param) in self.model.named_parameters():
            if not "classifier" in name:
                param.requires_grad = False

    def forward(self, x):
        y = self.model(x)
        return y.logits

    def to(self, device):
        self.model.to(device)
        return self

    def state_dict(self):
        # return self.model.state_dict()
        # if transfer learning, return only the classifier
        return self.model.classifier.state_dict()

    def load_state_dict(self, state_dict):
        # self.model.load_state_dict(state_dict)
        # if transfer learning, load only the classifier
        self.model.classifier.load_state_dict(state_dict)
    
    # @property
    def parameters(self):
        # return self.model.parameters()
        # if transfer learning, return only the classifier
        return self.model.classifier.parameters()

    def __call__(self, x):
        return self.forward(x)
    
    def named_parameters(self):
        # return self.model.named_parameters()
        # if transfer learning, return only the classifier
        return self.model.classifier.named_parameters()
    
    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()
