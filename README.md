# rdf2vec from SAREF Graph data and MLP training
 
This script can be used to create vector representations from a knowledge graph (KG) using rdf2vec. 
The KG is assumed to have followed the SAREF ontology [[1]](#SAREF) to describe series of measurements made by smart devices.
[pipeline_rdf2vec_mlp.ipynb](pipeline_rdf2vec_mlp.ipynb) provides a more indepth walkthrough.

The data in the graph consists of energy consumption from several smart devices, originally from the Household dataset from OPDS [[2]](#OPDS) and Historical Weather Data retrieved from World Weather Online.



## References
<a id="SAREF">[1]</a>
https://saref.etsi.org/core/v3.1.1/

<a id="OPDS">[2]</a>
Open Power System Data. 2020. Data Package Household Data. Version 2020-04-15.
https://data.open-power-system-data.org/household_data/2020-04-15/.
(Primary data from various sources, for a complete list see URL).

