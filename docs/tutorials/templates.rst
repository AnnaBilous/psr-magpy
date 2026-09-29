What a line stores
==================

``FieldLineTemplate`` is the list of quantities to store. ``FieldLine`` is one traced line, built with ``FieldLine.from_template``. The presets are:

- ``template_all``
- ``template_all_except_along_line``
- ``template_OFL_and_line_coordinates``
- ``template_OFL_calc``
- ``template_OFL_subset``
- ``template_line_coordinates_only``
- ``template_properties_along_line``
- ``template_surface_field_properties``

.. literalinclude:: ../../demo/pipeline/templates.py
   :language: python

.. code-block:: bash

   python demo/pipeline/templates.py

Each line is ``<preset> n_flags=<n>``. The last line is ``FieldLine n_enabled=5`` for ``template_OFL_calc``.
