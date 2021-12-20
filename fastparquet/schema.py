"""Utils for working with the parquet thrift models."""
from collections import OrderedDict

from . import parquet_thrift


def schema_tree(schema, i=0):
    root = schema[i]
    root["children"] = OrderedDict()
    while len(root["children"]) < root.num_children:
        i += 1
        s = schema[i]
        root["children"][s.name] = s
        if s.num_children not in [None, 0]:
            i = schema_tree(schema, i)
    if root.num_children:
        return i
    else:
        return i + 1


def schema_to_text(root, indent=[]):
    text = "".join(indent) + '- ' + root.name + ": "
    parts = []
    if root.type is not None:
        parts.append(parquet_thrift.Type._VALUES_TO_NAMES[root.type])
    if root.logicalType is not None:
        for key in dir(root.logicalType):
            if getattr(root.logicalType, key) is not None:
                if key == "TIMESTAMP":
                    unit = [k for k in dir(root.logicalType.TIMESTAMP.unit) if getattr(
                        root.logicalType.TIMESTAMP.unit, k) is not None][0]
                    parts.append(f"TIMESTAMP[{unit}]")
                else:
                    # extra parameters possible here
                    parts.append(key)
                break

    if root.converted_type is not None:
        parts.append(parquet_thrift.ConvertedType._VALUES_TO_NAMES[
                         root.converted_type])
    if root.repetition_type is not None:
        parts.append(parquet_thrift.FieldRepetitionType._VALUES_TO_NAMES[
                         root.repetition_type])
    text += ', '.join(parts)
    indent.append('|')
    if hasattr(root, 'children'):
        indent[-1] = '| '
        for i, child in enumerate(root["children"].values()):
            if i == len(root["children"]) - 1:
                indent[-1] = '  '
            text += '\n' + schema_to_text(child, indent)
    indent.pop()
    return text


def flatten(schema, root, name_parts=[]):
    if not hasattr(schema, 'children'):
        return
    if schema is not root:
        name_parts = name_parts + [schema.name]
    # root["children"].pop('.'.join(name_parts), None)
    for name, item in schema["children"].copy().items():
        if schema.repetition_type == parquet_thrift.FieldRepetitionType.REPEATED:
            continue
        if len(getattr(item, 'children', [])) == 0:
            root["children"]['.'.join(name_parts + [name])] = item
        elif item.converted_type in [parquet_thrift.ConvertedType.LIST,
                                     parquet_thrift.ConvertedType.MAP]:
            root["children"]['.'.join(name_parts + [name])] = item
        else:
            flatten(item, root, name_parts)
            item["isflat"] = True


class SchemaHelper(object):
    """Utility providing convenience methods for schema_elements."""

    def __init__(self, schema_elements):
        """Initialize with the specified schema_elements."""
        self.schema_elements = schema_elements
        for se in schema_elements:
            try:
                se.name = se.name.decode()
            except AttributeError:
                pass  # already a str
        self.root = schema_elements[0]
        self.schema_elements_by_name = dict(
            [(se.name, se) for se in schema_elements])
        schema_tree(schema_elements)
        self._text = None
        flatten(self.root, self.root)

    @property
    def text(self):
        if self._text is None:
            self._text = schema_to_text(self.schema_elements[0])
        return self._text

    def __eq__(self, other):
        return self.schema_elements == other.schema_elements

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.text

    def __repr__(self):
        return "<Parquet Schema with {} entries>".format(
            len(self.schema_elements))

    def schema_element(self, name):
        """Get the schema element with the given name or path"""
        root = self.root
        if isinstance(name, str):
            name = name.split('.')
        for part in name:
            root = root["children"][part]
        return root

    def is_required(self, name):
        """Return true if the schema element with the given name is required."""
        required = True
        if isinstance(name, str):
            name = name.split('.')
        parts = []
        for part in name:
            parts.append(part)
            s = self.schema_element(parts)
            if s.repetition_type != parquet_thrift.FieldRepetitionType.REQUIRED:
                required = False
                break
        return required

    def max_repetition_level(self, parts):
        """Get the max repetition level for the given schema path."""
        max_level = 0
        if isinstance(parts, str):
            parts = parts.split('.')
        for i in range(len(parts)):
            element = self.schema_element(parts[:i+1])
            if element.repetition_type == parquet_thrift.FieldRepetitionType.REPEATED:
                max_level += 1
        return max_level

    def max_definition_level(self, parts):
        """Get the max definition level for the given schema path."""
        max_level = 0
        if isinstance(parts, str):
            parts = parts.split('.')
        for i in range(len(parts)):
            element = self.schema_element(parts[:i+1])
            if element.repetition_type != parquet_thrift.FieldRepetitionType.REQUIRED:
                max_level += 1
        return max_level


def _optional(form, key=""):
    from awkward._v2 import forms
    return forms.ByteMaskedForm("i8", form, valid_when=True, form_key=key)


def _first(d):
    return list(d.values())[0]


def schema_to_awkward(se, topname=""):
    from .converted_types import simple, complex
    from awkward._v2 import forms

    stype = simple.get(se.type)
    ctype = complex.get(se.converted_type)
    name = se.name
    fullname = f"{topname}.{name}" if topname else name
    optional = se.repetition_type == parquet_thrift.FieldRepetitionType.OPTIONAL
    if not getattr(se, "children", False):
        # leaf node
        dtype = str(ctype or stype)
        if se.type in [parquet_thrift.Type.BYTE_ARRAY, parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY]:
            # string/bytes
            forms.NumpyForm("uint8")
            utf = se.converted_type == parquet_thrift.ConvertedType.UTF8
            parameters = {"__array__": ["bytestring", "string"][utf]}
            form = forms.ListOffsetForm(
                "i32",
                forms.NumpyForm("uint8"),
                parameters=parameters,
                form_key=fullname,
            )
        # TODO: other special cases: DECIMAL, etc.
        elif se.logicalType and se.logical_type.UNKNOWN is not None:
            # null
            form = forms.EmptyForm(form_key=fullname)
        elif dtype == "M8[ns]" and se.logical_type:
            # timestamp
            unit = se.logical_type.TIMESTAMP.unit
            if unit.MicroSeconds:
                dtype = "M8[us]"
            elif unit.MilliSeconds:
                dtype = "M8[ms]"
            form = forms.numpyform.NumpyForm(dtype, form_key=fullname)
        else:
            # any other number type
            form = forms.numpyform.NumpyForm(dtype, form_key=fullname)

    elif se.converted_type == parquet_thrift.ConvertedType.LIST:
        # list
        if _first(se.children).repetition_type == parquet_thrift.FieldRepetitionType.REPEATED:
            child = _first(_first(se.children).children)
            optional = True
        else:
            # required list type (rare)
            child = _first(se.children)
        form = forms.ListOffsetForm("i32", schema_to_awkward(child, topname=fullname), form_key=fullname)
    elif se.repetition_type == parquet_thrift.FieldRepetitionType.REPEATED:
        # a repeated field not marked as a LIST - probably doesn't happen
        raise NotImplementedError
    else:
        # not a leaf, not a list: must be a struct
        contents = [schema_to_awkward(child, topname=fullname) for child in se.children.values()]
        names = [child.name for child in se.children.values()]
        form = forms.RecordForm(contents, names)
    if optional:
        return _optional(form, key=fullname)
    else:
        return form


def _is_list_like(helper, column):
    if len(column.meta_data.path_in_schema) < 3:
        return False
    se = helper.schema_element(
        column.meta_data.path_in_schema[:-2])
    ct = se.converted_type
    if ct != parquet_thrift.ConvertedType.LIST:
        return False
    if len(se["children"]) > 1:
        return False
    se2 = list(se["children"].values())[0]
    if len(se2["children"]) > 1:
        return False
    if se2.repetition_type != parquet_thrift.FieldRepetitionType.REPEATED:
        return False
    se3 = list(se2["children"].values())[0]
    if se3.repetition_type == parquet_thrift.FieldRepetitionType.REPEATED:
        return False
    return True


def _is_map_like(helper, column):
    if len(column.meta_data.path_in_schema) < 3:
        return False
    se = helper.schema_element(
        column.meta_data.path_in_schema[:-2])
    ct = se.converted_type
    if ct != parquet_thrift.ConvertedType.MAP:
        return False
    if len(se["children"]) > 1:
        return False
    se2 = list(se["children"].values())[0]
    if len(se2["children"]) != 2:
        return False
    if se2.repetition_type != parquet_thrift.FieldRepetitionType.REPEATED:
        return False
    if set(se2["children"]) != {'key', 'value'}:
        return False
    se3 = se2["children"]['key']
    if se3.repetition_type != parquet_thrift.FieldRepetitionType.REQUIRED:
        return False
    se3 = se2["children"]['value']
    if se3.repetition_type == parquet_thrift.FieldRepetitionType.REPEATED:
        return False
    return True
