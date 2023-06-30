"""
A tool for putting things on top of other things.

This file contains the interface with Blender, while stacker.py contains the actual
stacking logic.
"""

# Blender info dict.
bl_info = {
    "name": "Stacker",
    "description": "A tool for putting things on top of other things.",
    "author": "",
    "location": "3D View > Add-ons",
    "warning": "",
    "category": "Development",
}

import sys
import pathlib

import numpy as np

import bpy
from mathutils import Vector

# Setup to let Blender see other .py files
if not dir in sys.path:
    parent = pathlib.Path(bpy.data.filepath).parent
    sys.path.append(str(parent))
    sys.path.append(str(parent / "scripts"))

import stacker


def get_bounds(obj):
    """
    A blender specific helper to get world min and max of an object's bounding box.

    Arguments:
        obj:    An instance of bpy.types.object.

    Returns:
        A np.array of shape (3, 2) where the first dimension represents spacial axes
        and the second dimension is min, max.
    """
    corners = np.asarray(
        [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    )

    mins = np.min(corners, axis=0, keepdims=True)
    maxs = np.max(corners, axis=0, keepdims=True)

    # resulting shape is (3, 2)
    bounds = np.concatenate([mins, maxs], axis=0).T

    for child in obj.children_recursive:
        other = get_bounds(child)
        bounds[:, 0] = np.minimum(bounds[:, 0], other[:, 0])
        bounds[:, 1] = np.maximum(bounds[:, 1], other[:, 1])

    return bounds


class StackerProperties(bpy.types.PropertyGroup):
    """
    Lists all the properties of this add-on.

    Arguments:
        centering:  Set this flag to center each object over the previous one.
        sorting:    Set this flag to sort objects by area (as viewed head-on from axis).
        padding:    An optional amount of space to put between each object.
        axis:       Axis with which to stack, ie. 2 will put objects on top of each
                    other vertically on the z axis.
    """

    centering: bpy.props.BoolProperty(name="Centering", description="", default=True)
    sorting: bpy.props.BoolProperty(name="Sort", description="", default=True)

    padding: bpy.props.FloatProperty(name="Padding", description="", default=0.0)

    axis: bpy.props.EnumProperty(
        name="Axis",
        description="Axis to stack on.",
        items=[
            ("X", "X", ""),
            ("Y", "Y", ""),
            ("Z", "Z", ""),
        ],
        default="Z",
    )


class StackerDropDownOperator(bpy.types.Operator):
    """
    The "drop down" operation.
    """

    bl_label = "Drop down"
    bl_idname = "wm.stackerdropdown"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        """
        Executor.
        """
        scene = context.scene
        objs = context.selected_objects
        active_object = context.active_object

        tool = scene.stacker_tool
        axis = ["X", "Y", "Z"].index(tool.axis)
        padding = tool.padding

        if len(objs) < 2:
            self.report({"WARNING"}, "At least two objects need to be selected.")
            return {"CANCELLED"}

        bounds = np.asarray([get_bounds(obj) for obj in objs])
        deltas = stacker.drop_down(bounds, axis, padding)

        for obj, delta in zip(objs, deltas):
            prev = stacker.translate_bounds(bounds, delta)
            obj.location += Vector(delta)

        return {"FINISHED"}


class StackerOperator(bpy.types.Operator):
    """
    The "stack" operation.
    """

    bl_label = "Stack"
    bl_idname = "wm.stacker"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        objs = context.selected_objects
        active_object = context.active_object

        tool = scene.stacker_tool
        axis = ["X", "Y", "Z"].index(tool.axis)
        padding = tool.padding
        centering = tool.centering
        sorting = tool.sorting

        if len(objs) < 2:
            self.report({"WARNING"}, "At least two objects need to be selected.")
            return {"CANCELLED"}

        if sorting:
            all_bounds = np.asarray([get_bounds(obj) for obj in objs])
            idxs = stacker.argsort_by_area(all_bounds, axis)
            idxs = idxs[::-1]  # Largest first

            # Re-order
            all_bounds = all_bounds[idxs]
            objs = [objs[i] for i in idxs]

            # Stack on top of the current position of the lowest object
            first_bounds = stacker.get_base(all_bounds, axis)
        else:
            # Put active object at the front
            objs.sort(key=lambda obj: obj != active_object)
            all_bounds = [get_bounds(obj) for obj in objs]

            # Stack on top of the first object
            first_bounds = all_bounds[0]

            # Don't move first object from where it currently is
            objs = objs[1:]
            all_bounds = all_bounds[1:]

        prev = first_bounds
        for obj, bounds in zip(objs, all_bounds):
            delta = stacker.stack_above(prev, bounds, axis, padding, centering)
            prev = stacker.translate_bounds(bounds, delta)
            obj.location += Vector(delta)

        return {"FINISHED"}


class StackerPanel(bpy.types.Panel):
    """
    Adds a panel to Blender's user interface.
    """

    bl_label = "Stacker"
    bl_idname = "OBJECT_PT_stacker"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Add-ons"
    bl_context = "objectmode"

    @classmethod
    def poll(self, context):
        """
        Blender poll function.
        """
        return context.object is not None

    def draw(self, context):
        """
        Blender draw function, adds widgets to layout.
        """
        layout = self.layout
        scene = context.scene
        tool = scene.stacker_tool

        layout.prop(tool, "axis")
        layout.prop(tool, "padding")
        layout.operator("wm.stackerdropdown")

        layout.prop(tool, "centering")
        layout.prop(tool, "sorting")
        layout.operator("wm.stacker")

        layout.separator()


classes = (
    StackerProperties,
    StackerOperator,
    StackerDropDownOperator,
    StackerPanel,
)


def register():
    """
    Blender register function.
    """
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.stacker_tool = bpy.props.PointerProperty(type=StackerProperties)


def unregister():
    """
    Unregister function called by Blender.
    """
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.stacker_tool


if __name__ == "__main__":
    register()
