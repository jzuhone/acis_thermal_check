================================
{{proc.name}} temperatures check
================================
.. role:: red

{% if proc.errors %}
Processing Errors
-----------------
.. class:: red
{% endif %}

{% if bsdir %}

Summary
--------         
.. class:: borderless

=====================  =============================================
Date start             {{proc.datestart}}
Date stop              {{proc.datestop}}
Model status           {%if any_viols %}:red:`NOT OK`{% else %}OK{% endif%} (Planning Limit = {{"%.1f"|format(proc.msid_limit)}} C)
{% if bsdir %}
Load directory         {{bsdir}}
{% endif %}
Run time               {{proc.run_time}} by {{proc.run_user}}
Run log                `<run.dat>`_
Temperatures           `<temperatures.dat>`_
States                 `<states.dat>`_
=====================  =============================================

{% for key in viols.keys() %}

{% if viols[key]["values"]|length > 0 %}
{{proc.msid}} {{viols[key]["name"]}} Violations
-----------------------------
=====================  =====================  ===================
Date start             Date stop              {{viols[key]["type"]}} Temperature
=====================  =====================  ===================
{% for viol in viols[key]["values"] %}
{{viol.datestart}}  {{viol.datestop}}  {{"%.2f"|format(viol.extemp)}}
{% endfor %}
=====================  =====================  ===================
{% else %}
No {{proc.msid}} {{viols[key]["name"]}} Violations
{% endif %}

{% endfor %}

.. image:: {{plots.default.filename}}
.. image:: {{plots.pow_sim.filename}}
.. image:: {{plots.roll.filename}}

{% endif %}

{% if not pred_only %}

==============================
{{proc.name}} Model Validation
==============================

MSID quantiles
---------------

Note: {{proc.name}} quantiles are calculated using only points where {{proc.msid}} > {{proc.hist_limit.0}} degC.

.. csv-table:: 
   :header: "MSID", "1%", "5%", "16%", "50%", "84%", "95%", "99%"
   :widths: 15, 10, 10, 10, 10, 10, 10, 10

{% for plot in plots_validation %}
{% if plot.quant01 %}
   {{plot.msid}},{{plot.quant01}},{{plot.quant05}},{{plot.quant16}},{{plot.quant50}},{{plot.quant84}},{{plot.quant95}},{{plot.quant99}}
{% endif %}
{% endfor%}

{% if valid_viols %}
Validation Violations
---------------------

.. csv-table:: 
   :header: "MSID", "Quantile", "Value", "Limit"
   :widths: 15, 10, 10, 10

{% for viol in valid_viols %}
   {{viol.msid}},{{viol.quant}},{{viol.value}},{{"%.2f"|format(viol.limit)}}
{% endfor%}

{% else %}
No Validation Violations
{% endif %}


{% for plot in plots_validation %}

{% if plot.msid == "ccd_count" %}

CCD/FEP Count
-------------

.. image:: {{plot.lines.filename}}

{% elif plot.msid == "earthheat__fptemp" %}

Earth Solid Angle
-----------------

.. image:: {{plot.lines.filename}}

{% else %}

{{ plot.msid }}
-----------------------

{% if plot.msid == proc.msid %}
{% if proc.hist_limit|length == 2 %}
Note: {{proc.name}} residual histograms include points where {{proc.msid}} {{proc.op.0}} {{proc.hist_limit.0}} degC in blue and points where {{proc.msid}} {{proc.op.1}} {{proc.hist_limit.1}} degC in red.
{% else %}
Note: {{proc.name}} residual histograms include only points where {{proc.msid}} {{proc.op.0}} {{proc.hist_limit.0}} degC.
{% endif %}
{% endif %}

.. image:: {{plot.lines.filename}}
.. image:: {{plot.hist.filename}}

{% endif %}

{% endfor %}

{% endif %}
