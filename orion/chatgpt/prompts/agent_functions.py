FUNCTIONS = [
    {
        "name": "dialog",
        "description": "talk to the user, usually the last function called for one turn",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "dialog content",
                }
            },
        },
        "required": ["content"],
    },
    {
        "name": "rotate",
        "description": "rotate the robot left or right",
        "parameters": {
            "type": "object",
            "properties": {
                "angle": {
                    "type": "number",
                    "description": "the angle degree to rotate the robot, > 0 for right, < 0 for left, should be in [-180, 180] and 15 degree multiples",
                },
                "detect_on": {
                    "type": "boolean",
                    "description": "whether using ego-view detection during the rotation",
                    "default": False,
                },
            },
        },
        "required": ["angle"],
    },
    {
        "name": "move",
        "description": "move the robot in the environment forward or backward",
        "parameters": {
            "type": "object",
            "properties": {
                "distance": {
                    "type": "number",
                    "description": "the total units to move the robot, > 0 for forward, < 0 for backward.",
                },
                "detect_on": {
                    "type": "boolean",
                    "description": "whether using ego-view detection during the rotation",
                    "default": False,
                },
            },
        },
        "required": ["distance"],
    },
    {
        "name": "goto_points",
        "description": "move the robot to specific points",
        "parameters": {
            "type": "object",
            "properties": {
                "points": {
                    "type": "array",
                    "description": "The list of points to go. Each point is a polar coordinate (distance, angle) tuple with respect to current robot position.",
                }
            },
        },
        "required": ["points"],
    },
    {
        "name": "goto_object",
        "description": "move the robot to a specific object.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": "The id of the object to go to. It is given by call `detect_object`, `retrieve_memory` or `search_object`",
                }
            },
        },
        "required": ["object_id"],
    },
    {
        "name": "change_view",
        "description": "go to another view point of the object",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": "The id of the object to change the view",
                }
            },
        },
        "required": ["object_id"],
    },
    {
        "name": "set_goal",
        "description": "set new object goal",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "The goal object name the user mentioned, such as bed, couch. But can not be floor and wall.",
                },
                "prompt": {
                    "type": "string",
                    "description": "more detailed phrase description of `target`. Be sure prompt contains the target",
                },
            },
        },
        "required": ["target", "prompt"],
    },
    {
        "name": "retrieve_memory",
        "description": "retrieve the memory to get hints where the object is, return <object_id, distance, angle, whether_in_same_room> tuple",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "The goal object name.",
                },
                "others": {
                    "type": "array",
                    "description": "Other related nearby objects that help the robot to find the target object.",
                },
            },
        },
        "required": ["target", "others"],
    },
    {
        "name": "update_memory",
        "description": "update the stored memory with the new information the user tells robot",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": "The id of the object to update the memory",
                },
                "pos_str_list": {
                    "type": "array",
                    "description": "The list of positive expression for the object, e.g. user says 'this is a bed', then the 'bed' is positive",
                },
                "neg_str_list": {
                    "type": "array",
                    "description": "The list of negative phrases for the the object, e.g. user says 'it's not a bed', then 'bed' is negative",
                },
            },
        },
        "required": ["object_id", "pos_str_list", "neg_str_list"],
    },
    {
        "name": "mask_object",
        "description": "if same type object found but not exact target object user wants, you can temporarily mask the object to avoid detect it again",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": "The id of the object to mask out for current user goal",
                },
            },
        },
        "required": ["object_id"],
    },
    {
        "name": "detect_object",
        "description": "detect the object using the ego-view detection model, return <object_id, distance, angle, whether_in_same_room> tuple",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "the target object you want to detect",
                },
                "prompt": {
                    "type": "string",
                    "description": "the detailed description contains the target",
                },
            },
        },
        "required": ["target", "prompt"],
    },
    {
        "name": "double_check",
        "description": "go to a new view point and detect the target again",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": "The id of the object to double check",
                },
            },
        },
        "required": ["object_id"],
    },
    {
        "name": "search_object",
        "description": "use the frontier-based exploration to search the object, return possible detected results. issue this command again can continue the searching",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "the target object you want to detect",
                },
                "prompt": {
                    "type": "string",
                    "description": "the detailed description contains the target",
                },
            },
        },
        "required": ["target", "prompt"],
    },
]
