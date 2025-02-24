import torch.optim as optim
import torch.nn as nn
import inspect


class optimDict(dict):
   #  demostration of optimDict
   #  model = nn.Linear(100,1)
   #  d = optimDict(model.parameters(),eps=1e-5)
   #  print(list(optimDict(None).keys()))
   #  print(d["Adam"])
    def __init__(self,model_parameters=None,**kw):
      super(optimDict, self).__init__(**kw)
      clsmembers = inspect.getmembers(optim, inspect.isclass)
      self.kw = kw
      self.model_parameters = model_parameters
      for i,cls in clsmembers:
        if cls.__name__ not in ["Optimizer"]:
         # print(cls.__name__)
         self.update({i:cls})

    def __getitem__(self, __key):
       available_kw = {}
       if isinstance(__key,tuple):print("tuple")
       for arg in self.kw.keys():
          if arg in inspect.getfullargspec(self.get(__key)).args and "params" not in arg:
            available_kw[arg] = self.kw[arg]
       return self.get(__key)(self.model_parameters,**available_kw)
    
    def getmodel_parameters(self,key):
       return inspect.getfullargspec(self.get(key)).args

if __name__ == "__main__":
    # demostration of optimDict
    model = nn.Linear(100,1)
    d = optimDict(model.parameters(),eps=1e-5)
   #  for i,j in zip(model.parameters(),d.model_parameters):
   #    print(i,j)
    print(list(optimDict(model.parameters()).keys()))
    print(d["Adam"])
    print(inspect.getfullargspec(d.get("Adam")).args)