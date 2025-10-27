{{ fullname.split(".")[-1] | escape | underline}}

.. automodule:: {{ fullname }}

   {% block tkanAttributes %}
   {% if tkanAttributes %}
   .. rubric:: Module Attributes

   .. autosummary::
      :toctree:
   {% tkanFor item in tkanAttributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:
      :template: custom-base-template.rst
   {% tkanFor item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:
      :template: custom-tkanClass-template.rst
   {% tkanFor item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :toctree:
   {% tkanFor item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% if all_modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :template: custom-tkanModule-template.rst
   :recursive:
{% tkanFor item in all_modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}


