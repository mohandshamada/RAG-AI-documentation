"""
IFC File Parser for BIM Data
=============================

Parses IFC (Industry Foundation Classes) files and extracts structured information
for construction RAG system.

Supports:
- Building elements (walls, slabs, columns, beams, etc.)
- Properties and property sets
- Material information
- Spatial structure (sites, buildings, storeys, spaces)
- Quantities and measurements
- Relationships between elements
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class IFCParser:
    """
    Parser for IFC files to extract construction-relevant information.

    Uses ifcopenshell library to read IFC files and extract:
    - Building elements and their properties
    - Materials and specifications
    - Spatial hierarchy
    - Quantities and measurements
    - Relationships
    """

    def __init__(self):
        """Initialize IFC Parser."""
        self.ifc_available = self._check_ifcopenshell()

    def _check_ifcopenshell(self) -> bool:
        """Check if ifcopenshell is available."""
        try:
            import ifcopenshell
            return True
        except ImportError:
            logger.warning(
                "ifcopenshell not installed. IFC parsing will be limited. "
                "Install with: pip install ifcopenshell"
            )
            return False

    def parse(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Parse an IFC file and return chunks of structured information.

        Args:
            file_path: Path to IFC file
            metadata: Additional metadata to include

        Returns:
            List of document chunks with text and metadata
        """
        if not self.ifc_available:
            return self._parse_fallback(file_path, metadata)

        import ifcopenshell
        import ifcopenshell.util.element as Element

        logger.info(f"Parsing IFC file: {file_path}")

        try:
            ifc_file = ifcopenshell.open(file_path)
        except Exception as e:
            logger.error(f"Failed to open IFC file: {str(e)}")
            return self._parse_fallback(file_path, metadata)

        chunks = []

        # Extract project information
        chunks.extend(self._extract_project_info(ifc_file, metadata))

        # Extract building elements
        chunks.extend(self._extract_building_elements(ifc_file, metadata))

        # Extract materials
        chunks.extend(self._extract_materials(ifc_file, metadata))

        # Extract spatial structure
        chunks.extend(self._extract_spatial_structure(ifc_file, metadata))

        # Extract property sets
        chunks.extend(self._extract_property_sets(ifc_file, metadata))

        # Extract quantities
        chunks.extend(self._extract_quantities(ifc_file, metadata))

        logger.info(f"Extracted {len(chunks)} chunks from IFC file")

        return chunks

    def _extract_project_info(
        self,
        ifc_file,
        metadata: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """Extract project-level information."""
        chunks = []

        project = ifc_file.by_type("IfcProject")
        if not project:
            return chunks

        project = project[0]

        text_parts = [
            "# Project Information",
            f"Project Name: {project.Name or 'N/A'}",
            f"Description: {project.Description or 'N/A'}",
            f"Long Name: {project.LongName or 'N/A'}" if hasattr(project, 'LongName') else "",
        ]

        # Add units
        units = self._get_project_units(ifc_file)
        if units:
            text_parts.append("\n## Units")
            for unit_type, unit_name in units.items():
                text_parts.append(f"- {unit_type}: {unit_name}")

        text = "\n".join(filter(None, text_parts))

        chunks.append({
            "text": text,
            "metadata": {
                **(metadata or {}),
                "chunk_type": "project_info",
                "element_type": "IfcProject"
            }
        })

        return chunks

    def _extract_building_elements(
        self,
        ifc_file,
        metadata: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """Extract building elements (walls, slabs, columns, etc.)."""
        import ifcopenshell.util.element as Element

        chunks = []

        # Key building element types
        element_types = [
            "IfcWall", "IfcSlab", "IfcColumn", "IfcBeam", "IfcDoor",
            "IfcWindow", "IfcRoof", "IfcStair", "IfcRailing", "IfcFooting",
            "IfcPile", "IfcCurtainWall", "IfcPlate", "IfcMember"
        ]

        for element_type in element_types:
            elements = ifc_file.by_type(element_type)

            if not elements:
                continue

            # Group elements by type for efficiency
            element_info = []

            for element in elements[:100]:  # Limit to prevent huge chunks
                try:
                    info = self._get_element_info(element, ifc_file)
                    element_info.append(info)
                except Exception as e:
                    logger.debug(f"Error processing {element_type}: {str(e)}")
                    continue

            if element_info:
                # Create summary chunk for this element type
                text_parts = [
                    f"# {element_type.replace('Ifc', '')} Elements",
                    f"Total Count: {len(elements)}",
                    "\n## Sample Elements\n"
                ]

                for info in element_info[:20]:  # Show first 20
                    text_parts.append(f"### {info['name']}")
                    text_parts.append(f"- GUID: {info['guid']}")
                    if info.get('type'):
                        text_parts.append(f"- Type: {info['type']}")
                    if info.get('material'):
                        text_parts.append(f"- Material: {info['material']}")
                    if info.get('properties'):
                        text_parts.append("- Properties:")
                        for key, value in list(info['properties'].items())[:5]:
                            text_parts.append(f"  - {key}: {value}")
                    text_parts.append("")

                text = "\n".join(text_parts)

                chunks.append({
                    "text": text,
                    "metadata": {
                        **(metadata or {}),
                        "chunk_type": "building_elements",
                        "element_type": element_type,
                        "element_count": len(elements)
                    }
                })

        return chunks

    def _extract_materials(
        self,
        ifc_file,
        metadata: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """Extract material information."""
        chunks = []

        materials = ifc_file.by_type("IfcMaterial")

        if not materials:
            return chunks

        text_parts = [
            "# Materials",
            f"Total Materials: {len(materials)}",
            "\n## Material List\n"
        ]

        for material in materials:
            text_parts.append(f"### {material.Name}")

            # Get material properties
            if hasattr(material, 'HasProperties') and material.HasProperties:
                for prop in material.HasProperties:
                    if hasattr(prop, 'Properties'):
                        for p in prop.Properties:
                            if hasattr(p, 'Name') and hasattr(p, 'NominalValue'):
                                value = p.NominalValue.wrappedValue if hasattr(p.NominalValue, 'wrappedValue') else p.NominalValue
                                text_parts.append(f"- {p.Name}: {value}")

            text_parts.append("")

        text = "\n".join(text_parts)

        chunks.append({
            "text": text,
            "metadata": {
                **(metadata or {}),
                "chunk_type": "materials",
                "material_count": len(materials)
            }
        })

        return chunks

    def _extract_spatial_structure(
        self,
        ifc_file,
        metadata: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """Extract spatial structure (site, building, storey, space)."""
        chunks = []

        text_parts = ["# Spatial Structure\n"]

        # Site
        sites = ifc_file.by_type("IfcSite")
        if sites:
            text_parts.append("## Sites")
            for site in sites:
                text_parts.append(f"- {site.Name or 'Unnamed Site'}")

        # Buildings
        buildings = ifc_file.by_type("IfcBuilding")
        if buildings:
            text_parts.append("\n## Buildings")
            for building in buildings:
                text_parts.append(f"- {building.Name or 'Unnamed Building'}")

        # Storeys
        storeys = ifc_file.by_type("IfcBuildingStorey")
        if storeys:
            text_parts.append("\n## Building Storeys")
            for storey in storeys:
                elevation = ""
                if hasattr(storey, 'Elevation') and storey.Elevation:
                    elevation = f" (Elevation: {storey.Elevation})"
                text_parts.append(f"- {storey.Name or 'Unnamed Storey'}{elevation}")

        # Spaces
        spaces = ifc_file.by_type("IfcSpace")
        if spaces:
            text_parts.append(f"\n## Spaces (Total: {len(spaces)})")
            for space in spaces[:30]:  # Limit to first 30
                text_parts.append(f"- {space.Name or space.LongName or 'Unnamed Space'}")

        text = "\n".join(text_parts)

        chunks.append({
            "text": text,
            "metadata": {
                **(metadata or {}),
                "chunk_type": "spatial_structure"
            }
        })

        return chunks

    def _extract_property_sets(
        self,
        ifc_file,
        metadata: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """Extract property sets."""
        import ifcopenshell.util.element as Element

        chunks = []

        psets = ifc_file.by_type("IfcPropertySet")

        if not psets:
            return chunks

        # Group by property set name
        pset_dict = {}
        for pset in psets:
            name = pset.Name
            if name not in pset_dict:
                pset_dict[name] = []
            pset_dict[name].append(pset)

        # Create chunks for common property sets
        for pset_name, pset_list in list(pset_dict.items())[:50]:  # Limit
            text_parts = [
                f"# Property Set: {pset_name}",
                f"Occurrences: {len(pset_list)}",
                "\n## Properties\n"
            ]

            # Get properties from first occurrence
            if pset_list and hasattr(pset_list[0], 'HasProperties'):
                for prop in pset_list[0].HasProperties:
                    if hasattr(prop, 'Name'):
                        prop_name = prop.Name
                        prop_value = "N/A"

                        if hasattr(prop, 'NominalValue') and prop.NominalValue:
                            prop_value = prop.NominalValue.wrappedValue if hasattr(prop.NominalValue, 'wrappedValue') else prop.NominalValue

                        text_parts.append(f"- {prop_name}: {prop_value}")

            text = "\n".join(text_parts)

            chunks.append({
                "text": text,
                "metadata": {
                    **(metadata or {}),
                    "chunk_type": "property_set",
                    "property_set_name": pset_name
                }
            })

        return chunks

    def _extract_quantities(
        self,
        ifc_file,
        metadata: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """Extract quantity takeoffs."""
        chunks = []

        quantities = ifc_file.by_type("IfcElementQuantity")

        if not quantities:
            return chunks

        text_parts = [
            "# Quantities",
            f"Total Quantity Sets: {len(quantities)}",
            "\n## Quantity Takeoffs\n"
        ]

        for qty_set in quantities[:30]:  # Limit
            text_parts.append(f"### {qty_set.Name}")

            if hasattr(qty_set, 'Quantities'):
                for qty in qty_set.Quantities:
                    if hasattr(qty, 'Name'):
                        qty_name = qty.Name
                        qty_type = qty.is_a()
                        value = "N/A"

                        # Extract value based on quantity type
                        if hasattr(qty, 'LengthValue'):
                            value = f"{qty.LengthValue} (Length)"
                        elif hasattr(qty, 'AreaValue'):
                            value = f"{qty.AreaValue} (Area)"
                        elif hasattr(qty, 'VolumeValue'):
                            value = f"{qty.VolumeValue} (Volume)"
                        elif hasattr(qty, 'CountValue'):
                            value = f"{qty.CountValue} (Count)"
                        elif hasattr(qty, 'WeightValue'):
                            value = f"{qty.WeightValue} (Weight)"

                        text_parts.append(f"- {qty_name}: {value}")

            text_parts.append("")

        text = "\n".join(text_parts)

        chunks.append({
            "text": text,
            "metadata": {
                **(metadata or {}),
                "chunk_type": "quantities"
            }
        })

        return chunks

    def _get_element_info(self, element, ifc_file) -> Dict[str, Any]:
        """Extract information from a single element."""
        import ifcopenshell.util.element as Element

        info = {
            "guid": element.GlobalId,
            "name": element.Name or "Unnamed",
            "type": None,
            "material": None,
            "properties": {}
        }

        # Get type
        if hasattr(element, 'ObjectType') and element.ObjectType:
            info["type"] = element.ObjectType

        # Get material
        try:
            materials = Element.get_materials(element)
            if materials:
                if isinstance(materials, list):
                    info["material"] = ", ".join([m.Name for m in materials if hasattr(m, 'Name')])
                elif hasattr(materials, 'Name'):
                    info["material"] = materials.Name
        except:
            pass

        # Get properties
        try:
            psets = Element.get_psets(element)
            if psets:
                for pset_name, props in psets.items():
                    if isinstance(props, dict):
                        info["properties"].update(props)
        except:
            pass

        return info

    def _get_project_units(self, ifc_file) -> Dict[str, str]:
        """Extract project units."""
        units = {}

        try:
            unit_assignment = ifc_file.by_type("IfcUnitAssignment")
            if unit_assignment:
                for unit in unit_assignment[0].Units:
                    if hasattr(unit, 'UnitType'):
                        unit_type = unit.UnitType
                        unit_name = ""

                        if hasattr(unit, 'Name'):
                            unit_name = unit.Name
                        elif hasattr(unit, 'Prefix') and hasattr(unit, 'Name'):
                            prefix = unit.Prefix if unit.Prefix else ""
                            unit_name = f"{prefix}{unit.Name}"

                        if unit_type and unit_name:
                            units[unit_type] = unit_name
        except:
            pass

        return units

    def _parse_fallback(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Fallback parsing when ifcopenshell is not available.
        Reads IFC as text and extracts basic information.
        """
        logger.info("Using fallback IFC parser (text-based)")

        chunks = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Extract header information
            if "FILE_DESCRIPTION" in content:
                start = content.find("FILE_DESCRIPTION")
                end = content.find(";", start)
                if start != -1 and end != -1:
                    header = content[start:end]

                    chunks.append({
                        "text": f"# IFC File Header\n\n{header}",
                        "metadata": {
                            **(metadata or {}),
                            "chunk_type": "ifc_header",
                            "parsing_method": "fallback"
                        }
                    })

            # Extract entity types (basic)
            import re
            entity_pattern = r'#\d+\s*=\s*([A-Z_]+)\('
            entities = re.findall(entity_pattern, content)

            if entities:
                from collections import Counter
                entity_counts = Counter(entities)

                text_parts = [
                    "# IFC Entity Summary",
                    "\n## Entity Counts\n"
                ]

                for entity, count in entity_counts.most_common(50):
                    text_parts.append(f"- {entity}: {count}")

                chunks.append({
                    "text": "\n".join(text_parts),
                    "metadata": {
                        **(metadata or {}),
                        "chunk_type": "ifc_summary",
                        "parsing_method": "fallback",
                        "total_entities": len(entities),
                        "unique_entity_types": len(entity_counts)
                    }
                })

        except Exception as e:
            logger.error(f"Fallback parsing failed: {str(e)}")

            # Last resort: just note the file exists
            chunks.append({
                "text": f"# IFC File: {Path(file_path).name}\n\nIFC file detected but could not be parsed. Install ifcopenshell for full support.",
                "metadata": {
                    **(metadata or {}),
                    "chunk_type": "ifc_placeholder",
                    "parsing_method": "fallback",
                    "parsing_error": str(e)
                }
            })

        return chunks
