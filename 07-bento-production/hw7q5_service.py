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
modBenML = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")

print('>> Run model <<')

modBenMLRun = modBenML.to_runner()

svc = bentoml.Service("hw7q5_service", runners=[modBenMLRun])

@svc.api(input=NumpyNdarray(), output=JSON())

async def classify(UserProfile):
    print('>> in classify <<')
    print('>> UserProfile :' , UserProfile)
    prd = await modBenMLRun.predict.async_run(UserProfile)
    print('>> prd = ' , prd)
    return( { "prediction" : prd  })
