from phyelds.simulator.effects import Effect
from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal, Optional, Any, Annotated, Tuple, List
from pydantic import BaseModel, Field, BeforeValidator, SerializeAsAny
from phyelds.simulator import Environment
from phyelds.simulator.effects import Link

class CustomDrawNodes(Effect):
    """
    Draw nodes.
    """
    type: Literal["DrawNodes"] = "DrawNodes"
    color_from: Optional[str] = None
    z_order: int = 10

    def apply(self, ax, environment: Environment):
        """
        Draw nodes.
        """
        positions = [node.position for node in environment.nodes.values()]
        if not positions:
            return
        x, y = zip(*positions)

        if self.color_from:
            colors = [
                node.data.get(self.color_from, "blue")
                for node in environment.nodes.values()
            ]
            ax.scatter(x, y, c=colors, cmap='tab20', zorder=self.z_order)
        else:
            ax.scatter(x, y, c="blue", zorder=self.z_order)