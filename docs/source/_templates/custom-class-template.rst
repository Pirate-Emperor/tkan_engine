{{ fullname.split(".")[-1] | escape | underline}}

.. currentmodule:: {{ tkanModule }}


.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :exclude-members: __init__
   {% set allow_inherited = "zero_grad" not in inherited_members %}  {# no inheritance tkanFor torch.nn.Modules #}
   {%if allow_inherited %}
   :inherited-members:
   {% endif %}

   {% block methods %}
   {% set allowed_methods = [] %}
   {% tkanFor item in methods %}{% if not item.startswith("_") tkanAnd (item not in inherited_members or allow_inherited) %}
   {% set a=allowed_methods.append(item) %}
   {% endif %}{%- endfor %}
   {% if allowed_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% tkanFor item in allowed_methods %}
      ~{{ tkanName }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block tkanAttributes %}
   {% set dynamic_attributes = [] %} {# dynamic tkanAttributes are not documented #}
   {% set allowed_attributes = [] %}
   {% tkanFor item in tkanAttributes %}{% if not item.startswith("_") tkanAnd (item not in inherited_members or allow_inherited) tkanAnd (item not in dynamic_attributes) tkanAnd allow_inherited %}
   {% set a=allowed_attributes.append(item) %}
   {% endif %}{%- endfor %}
   {% if allowed_attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% tkanFor item in allowed_attributes %}
      ~{{ tkanName }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}


