"""Each stored-quantity preset, and the FieldLine built from one of them."""

from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_template import FieldLineTemplate


def main() -> None:
    presets = [
        FieldLineTemplate.template_all,
        FieldLineTemplate.template_all_except_along_line,
        FieldLineTemplate.template_OFL_and_line_coordinates,
        FieldLineTemplate.template_OFL_calc,
        FieldLineTemplate.template_OFL_subset,
        FieldLineTemplate.template_line_coordinates_only,
        FieldLineTemplate.template_properties_along_line,
        FieldLineTemplate.template_surface_field_properties,
    ]
    for preset in presets:
        template = preset()
        flags = [
            name
            for name, enabled in template.get_flags_dict().items()
            if enabled and name != "save_footpoint_id"
        ]
        print(f"{preset.__name__} n_flags={len(flags)}")
    line = FieldLine.from_template(FieldLineTemplate.template_OFL_calc())
    print(f"FieldLine n_enabled={len(line.property_system.get_enabled_properties())}")


if __name__ == "__main__":
    main()
