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
Model status           {%if any_viols %}:red:`NOT OK`{% else %}OK{% endif%}
{% if bsdir %}
Load directory         {{bsdir}}
{% endif %}
Run time               {{proc.run_time}} by {{proc.run_user}}
Run log                `<run.dat>`_
Temperatures           `<temperatures.dat>`_
{% if proc.msid == "FPTEMP" %}
Earth Solid Angles     `<earth_solid_angles.dat>`_
{% endif %}
States                 `<states.dat>`_
=====================  =============================================

{% for key in viols.keys() %}

{% if viols[key]["values"]|length > 0 %}
{{proc.msid}} {{viols[key]["name"]}} Violations
---------------------------------------------------
{% if proc.msid == "FPTEMP" %}
=====================  =====================  =================  ==================  ==================
Date start             Date stop              Duration (ks)      Max temperature     Obsids
=====================  =====================  =================  ==================  ==================
{% for viol in viols[key]["values"] %}
{{viol.datestart}}  {{viol.datestop}}  {{"{:3.2f}".format(viol.duration).rjust(8)}}            {{"%.2f"|format(viol.extemp)}}             {{viol.obsid}}
{% endfor %}
=====================  =====================  =================  ==================  ==================
{% else %}
=====================  =====================  =================  ===================
Date start             Date stop              Duration (ks)      {{viols[key]["type"]}} Temperature
=====================  =====================  =================  ===================
{% for viol in viols[key]["values"] %}
{{viol.datestart}}  {{viol.datestop}}  {{"{:3.2f}".format(viol.duration).rjust(8)}}           {{"{:.2f}".format(viol.extemp)}}
{% endfor %}
=====================  =====================  =================  ===================
{% endif %}
{% else %}
No {{proc.msid}} {{viols[key]["name"]}} Violations
{% endif %}

{% endfor %}

.. image:: {{plots.default.filename}}
.. image:: {{plots.pow_sim.filename}}
{% if proc.msid == "FPTEMP" %}
.. image:: {{plots.roll_taco.filename}}
{% else %}
.. image:: {{plots.roll.filename}}
{% endif %}

{% endif %}

{% if not pred_only %}

==============================
{{proc.name}} Model Validation
==============================

MSID quantiles
---------------

{% if proc.msid == "FPTEMP" %}
{% set quan_text = proc.hist_limit.0|join(" C <= FPTEMP <= ") + " C" %}
{% else %}
{% set quan_text = proc.msid + " >= " ~ proc.hist_limit.0 + " C" %}
{% endif %}

Note: Quantiles are calculated using only points where {{quan_text}}.

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
{% if proc.msid == "FPTEMP" %}
{% set hist_string = proc.hist_limit.0|join(" C <= FPTEMP <= ") + " C" %}
{% elif proc.hist_limit|length == 2 %}
{% set hist_string = proc.msid + " " ~ proc.op.0 + " " ~ proc.hist_limit.0 + " C in blue and points where " ~ proc.msid + " " ~ proc.op.1 + " " ~ proc.hist_limit.1 + " C in red" %}
{% else %}
{% set hist_string = proc.msid + " " ~ proc.op.0 + " " ~ proc.hist_limit.0 + " C" %}
{% endif %}
Note: {{proc.msid}} residual histograms include only points where {{hist_string}}.
{% endif %}

.. image:: {{plot.lines.filename}}
.. image:: {{plot.hist.filename}}

{% endif %}

{% endfor %}

{% if proc.msid == "FPTEMP" %}

ADDITIONAL PLOTS
-----------------------

Additional plots of FPTEMP vs TIME for different temperature ranges

.. image:: fptempM120toM119.png
.. image:: fptempM120toM90.png

{% endif %}

{% endif %}
