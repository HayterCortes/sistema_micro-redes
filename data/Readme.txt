Los datos eléctricos deben ser premultiplicados antes de ser procesados, para generar el
mismo caso de estudio de la tesis de magíster. Además incluyo los nombres de los archivos
y qué variable representan:

MG1
Demanda eléctrica: winter_30D, se multiplica por 1
Generación: pv_wint, se multiplica por 22
Demanda hídrica: Dwellings30Water, se multiplica por 1

MG2
Demanda eléctrica: winter_60D, se multiplica por 1
Generación: wind_inv, se multiplica por 8.49
Demanda hídrica: Dwellings60Water, se multiplica por 1

MG3
Demanda eléctrica: School_inv, se multiplica por 0.45
Generación: Combinar generación MG1 x30 + generación MG2 x5
Demanda hídrica: SchoolWater, se multiplica por 1

Estos datos tienen un muestreo de 1 minuto.