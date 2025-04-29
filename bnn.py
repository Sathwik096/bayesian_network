
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([('rain','wetgrass')])
cpd_rain = TabularCPD(variable='rain', variable_card=2, 
                       values=[[0.7], [0.3]])  
cpd_wetgrass = TabularCPD(variable='wetgrass', variable_card=2, 
                          values=[[0.9,0.2],[0.1,0.8]],
 
                         evidence=['rain'], 
                         evidence_card=[2])
model.add_cpds(cpd_rain, cpd_wetgrass)
print(model.check_model())
infer = VariableElimination(model)

result = infer.query(variables=['rain'], evidence={'wetgrass': 1})
print(result)
