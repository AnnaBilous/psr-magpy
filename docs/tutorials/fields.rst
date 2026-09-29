Field types
===========

A component is a dipole or a quadrupole. ``DipoleFieldComponent`` and ``QuadrupoleFieldComponent`` are the field objects. ``MultipoleField`` holds them. ``FieldConfig`` and ``FieldComponentConfig`` are the same components as experiment input. ``moment_type`` is ``"dipole"`` or ``"quadrupole"``.

.. literalinclude:: ../../demo/pipeline/fields.py
   :language: python

.. code-block:: bash

   python demo/pipeline/fields.py

The run prints ``components=['dipole', 'quadrupole']``.
