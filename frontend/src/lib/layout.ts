import dagre from "dagre";

import type { TraceEdge, TraceNode } from "@/types/trace";

export const NODE_W = 260;
export const NODE_H = 80;

// Spacing follows the card size, so a resized card keeps the picture in proportion.
const GAP = NODE_H / 2;
const RANK_GAP = NODE_W / 2;
const MARGIN = 32;
// Edges the question walked pull harder, which keeps its path short and straight.
const WALKED_EDGE_WEIGHT = 4;
const MIN_SHELF_COLUMNS = 3;

type Point = { x: number; y: number };

/** Top-left corners for every node that has an edge, laid out left to right. */
function placeLinked(nodes: TraceNode[], edges: TraceEdge[]): Map<string, Point> {
  const placed = new Map<string, Point>();
  if (edges.length === 0) return placed;

  const graph = new dagre.graphlib.Graph();
  graph.setGraph({ rankdir: "LR", nodesep: GAP, ranksep: RANK_GAP, marginx: MARGIN, marginy: MARGIN });
  graph.setDefaultEdgeLabel(() => ({}));
  const ends = new Set(edges.flatMap((edge) => [edge.source, edge.target]));
  nodes
    .filter((node) => ends.has(node.id))
    .forEach((node) => graph.setNode(node.id, { width: NODE_W, height: NODE_H }));
  edges.forEach((edge) =>
    graph.setEdge(edge.source, edge.target, { weight: edge.active ? WALKED_EDGE_WEIGHT : 1, minlen: 1 }),
  );
  dagre.layout(graph);

  for (const id of graph.nodes()) {
    const centre = graph.node(id);
    placed.set(id, { x: Math.round(centre.x - NODE_W / 2), y: Math.round(centre.y - NODE_H / 2) });
  }
  return placed;
}

/** Nodes without edges, shelved in rows underneath, as wide as the graph above. */
function shelve(loose: TraceNode[], above: Map<string, Point>): Map<string, Point> {
  const corners = [...above.values()];
  const width = corners.reduce((widest, corner) => Math.max(widest, corner.x + NODE_W), 0);
  const top = corners.reduce((lowest, corner) => Math.max(lowest, corner.y + NODE_H + RANK_GAP / 2), MARGIN);
  const pitch = { x: NODE_W + GAP, y: NODE_H + GAP };
  const columns = Math.max(MIN_SHELF_COLUMNS, Math.floor(width / pitch.x));

  const shelved = new Map<string, Point>();
  loose.forEach((node, slot) => {
    shelved.set(node.id, {
      x: MARGIN + (slot % columns) * pitch.x,
      y: top + Math.floor(slot / columns) * pitch.y,
    });
  });
  return shelved;
}

export function layoutGraph(
  nodes: TraceNode[],
  edges: TraceEdge[],
): { nodes: TraceNode[]; edges: TraceEdge[] } {
  const known = new Set(nodes.map((node) => node.id));
  const drawable = edges.filter((edge) => known.has(edge.source) && known.has(edge.target));

  const linked = placeLinked(nodes, drawable);
  const shelved = shelve(nodes.filter((node) => !linked.has(node.id)), linked);

  return {
    nodes: nodes.map((node) => ({
      ...node,
      position: linked.get(node.id) ?? shelved.get(node.id) ?? node.position,
      orphan: !linked.has(node.id),
    })),
    edges: drawable,
  };
}
