import dagre from "dagre";

import type { TraceEdge, TraceNode } from "@/types/trace";

export const NODE_W = 260;
export const NODE_H = 80;

const GRID_GAP_X = 48;
const GRID_GAP_Y = 40;
const MARGIN = 40;
const ORPHAN_OFFSET = 80;

export function layoutGraph(
  nodes: TraceNode[],
  edges: TraceEdge[],
): { nodes: TraceNode[]; edges: TraceEdge[] } {
  const present = new Set(nodes.map((node) => node.id));
  const kept = edges.filter((edge) => present.has(edge.source) && present.has(edge.target));

  const linked = new Set<string>();
  for (const edge of kept) {
    linked.add(edge.source);
    linked.add(edge.target);
  }

  const graph = new dagre.graphlib.Graph();
  graph.setGraph({
    rankdir: "LR",
    nodesep: 48,
    ranksep: 120,
    marginx: 40,
    marginy: 40,
    ranker: "network-simplex",
  });
  graph.setDefaultEdgeLabel(() => ({}));

  for (const node of nodes) {
    if (linked.has(node.id)) graph.setNode(node.id, { width: NODE_W, height: NODE_H });
  }
  for (const edge of kept) {
    graph.setEdge(edge.source, edge.target, { weight: edge.active ? 8 : 1 });
  }
  if (linked.size > 0) dagre.layout(graph);

  const positions = new Map<string, { x: number; y: number }>();
  let bottom = 0;
  for (const id of linked) {
    const placed = graph.node(id);
    const x = Math.round(placed.x - NODE_W / 2);
    const y = Math.round(placed.y - NODE_H / 2);
    positions.set(id, { x, y });
    bottom = Math.max(bottom, y + NODE_H);
  }

  const orphans = nodes.filter((node) => !linked.has(node.id));
  const columns = Math.max(1, Math.ceil(Math.sqrt(orphans.length)));
  const startY = linked.size > 0 ? bottom + ORPHAN_OFFSET : MARGIN;
  orphans.forEach((node, index) => {
    positions.set(node.id, {
      x: MARGIN + (index % columns) * (NODE_W + GRID_GAP_X),
      y: startY + Math.floor(index / columns) * (NODE_H + GRID_GAP_Y),
    });
  });

  return {
    nodes: nodes.map((node) => ({
      ...node,
      position: positions.get(node.id) ?? node.position,
      orphan: !linked.has(node.id),
    })),
    edges: kept,
  };
}
