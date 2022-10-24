import numpy as np

import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray

from pydantic import BaseModel

print('>> Import completed <<')

class UserProfile(BaseModel):
    name: str
    age: int
    country: str
    rating: float


print('>> Get model <<')

# q5_coolmodel.bentomodel
#modBnt = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")

# q6_coolmodel2.bentomodel
modBenML = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")

print('>> Run mmodel <<')

modBenMLRun = modBenML.to_runner()


svc = bentoml.Service("hw7q6_service", runners=[modBenMLRun])

@svc.api(input=NumpyNdarray(), output=JSON())

def classify(UserProfile):
    print('>> in classify <<')
    print('> UserProfile :' , UserProfile)
    prd = modBenMLRun.predict.run(UserProfile)
    print('> prd = ' , prd)
    return( { "prediction" : prd  })