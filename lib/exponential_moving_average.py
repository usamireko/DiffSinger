"""
Copyright © 2025 https://github.com/autumn-DL/Basic_Module

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
class ExponentialMovingAverageV2:
    def __init__(self, model, decay=0.99, ignored_layers=None):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.ignored_layers = ignored_layers
        self.lock = False
        self.key_list = []

    def register(self):
        for name, param in self.model.named_parameters():
            skip = 0
            if self.ignored_layers is not None:
                for b in self.ignored_layers:
                    if name.startswith(b):
                        skip = 1
                        break
            if skip == 1:
                continue
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                self.key_list.append(name)

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

    def save_state_dict(self):
        return self.shadow

    def update(self):

        model_p = self.model.named_parameters()
        model_p = dict(model_p)
        keys = list(model_p.keys())
        device = (model_p[keys[0]]).device
        for key in self.key_list:
            new_average = (1.0 - self.decay) * (model_p[key].data).to(device) + self.decay * (self.shadow[key]).to(
                device)
            self.shadow[key] = new_average.clone()

    def apply_shadow(self):

        model_p = self.model.named_parameters()
        model_p = dict(model_p)
        keys = list(model_p.keys())
        device = (model_p[keys[0]]).device
        for key in self.key_list:
            self.backup[key] = (model_p[key].data).to(device)
            model_p[key].data = (self.shadow[key]).to(device)

    def restore(self):
        model_p = self.model.named_parameters()
        model_p = dict(model_p)
        keys = list(model_p.keys())
        device = (model_p[keys[0]]).device
        for key in self.key_list:
            model_p[key].data = (self.backup[key]).to(device)

        self.backup = {}
