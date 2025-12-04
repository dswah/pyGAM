..  class.rst

{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}
.. autoclass:: {{ objname }}
    :members:
    :show-inheritance:
    :inherited-members:
    {% block methods %}
        {% if methods %}
            .. rubric:: {{ _('Methods') }}
            .. autosummary::
                :nosignatures:
                {% for item in methods %}
                    ~{{ name }}.{{ item }}
                {%- endfor %}
        {% endif %}
    {% endblock %}
