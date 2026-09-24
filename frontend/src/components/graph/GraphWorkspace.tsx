import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Minus, Network, Plus, Scan } from "lucide-react";
import ReactFlow, {
  Background, BackgroundVariant, Handle, MarkerType, Panel, Position, ReactFlowProvider,
  useEdgesState, useNodesState, useReactFlow,
  type Edge, type Node, type NodeProps, type OnNodesChange,
} from "reactflow";

import { EntityIcon } from "@/components/graph/EntityIcon";
import { NodeInspector } from "@/components/graph/NodeInspector";
import { layoutGraph, NODE_H, NODE_W } from "@/lib/layout";
import { cn } from "@/lib/utils";
import type { TraceNode, TraceState } from "@/types/trace";

type FlowNode = Node<TraceNode>;

export interface CitationFocusRequest { nodeId: string; nonce: number }

interface GraphWorkspaceProps {
  trace: TraceState;
  activeGraphId?: string;
  citationFocus?: CitationFocusRequest;
  summarize?: (key: string, text: string) => Promise<{ summary: string; error?: string | null }>;
}

function EntityNode({ data, selected }: NodeProps<TraceNode>) {
  return (
    <div className={cn("entity-node", data.active ? "entity-node-active" : "entity-node-context", selected && "entity-node-selected")}>
      <Handle type="target" position={Position.Left} />
      {data.score !== undefined ? <span className="entity-score">{data.score.toFixed(3)}</span> : null}
      <span className="entity-icon"><EntityIcon type={data.type} /></span>
      <span className="min-w-0">
        <span className="block truncate text-sm font-semibold text-ink">{data.label}</span>
        <span className="mt-1 block truncate text-xs text-ink-dim">{data.meta?.subtitle ?? data.type}</span>
      </span>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const nodeTypes = { entity: EntityNode };

type Graph = TraceState["graph"];

// What the question touched: nodes it ranked or walked through, and the edges
// it walked. Neighbours pulled in only to give those nodes context are left
// out unless asked for; around a busy node they are most of the picture.
function questionGraph(graph: Graph, showNeighbours: boolean): Graph {
  const touched = graph.nodes.filter((node) => node.active);
  if (showNeighbours || touched.length === 0) return graph;
  const ids = new Set(touched.map((node) => node.id));
  return {
    nodes: touched,
    edges: graph.edges.filter((edge) => edge.active && ids.has(edge.source) && ids.has(edge.target)),
  };
}

function toFlowNodes(graph: Graph): FlowNode[] {
  const laidOut = layoutGraph(graph.nodes, graph.edges);
  return laidOut.nodes.map((node) => ({
    id: node.id, type: "entity", position: node.position, data: node,
    sourcePosition: Position.Right, targetPosition: Position.Left,
    style: { width: NODE_W, height: NODE_H },
  }));
}

function toFlowEdges(graph: Graph): Edge[] {
  return graph.edges.map((edge) => ({
    id: edge.id, source: edge.source, target: edge.target, type: "step",
    label: edge.relation?.toLowerCase().replace(/_/g, " "),
    labelStyle: { fill: "rgb(var(--m-ink-dim))", fontFamily: "ui-monospace, monospace", fontSize: 11 },
    labelBgStyle: { fill: "rgb(var(--m-paper))", fillOpacity: 0.9 },
    labelBgPadding: [5, 3] as [number, number],
    markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16, color: edge.active ? "rgb(var(--m-blue))" : "rgb(var(--m-ink-muted))" },
    animated: edge.active, className: edge.active ? "edge-active" : undefined,
    style: { stroke: edge.active ? "rgb(var(--m-blue))" : "rgb(var(--m-ink-muted))", strokeWidth: 1.5 },
  }));
}

interface NeighbourToggle { shown: boolean; hidden: number; onToggle: () => void }

function CanvasControls({ neighbours }: { neighbours: NeighbourToggle }) {
  const { fitView, zoomIn, zoomOut } = useReactFlow();
  const label = neighbours.shown ? "Hide neighbours" : `Show ${neighbours.hidden} neighbours`;
  return (
    <Panel position="top-right" className="!m-4 flex gap-2">
      {neighbours.shown || neighbours.hidden > 0 ? (
        <button
          type="button" className={cn("canvas-control relative", neighbours.shown && "text-blue")}
          onClick={neighbours.onToggle} aria-label={label} aria-pressed={neighbours.shown} title={label}
        >
          <Network size={17} />
          {!neighbours.shown ? (
            <span className="absolute -right-1.5 -top-1.5 min-w-[18px] rounded-full bg-blue px-1 text-center font-mono text-[10px] leading-[18px] text-white">
              {neighbours.hidden > 99 ? "99+" : neighbours.hidden}
            </span>
          ) : null}
        </button>
      ) : null}
      <button type="button" className="canvas-control" onClick={() => void zoomIn({ duration: 180 })} aria-label="Zoom in"><Plus size={17} /></button>
      <button type="button" className="canvas-control" onClick={() => void zoomOut({ duration: 180 })} aria-label="Zoom out"><Minus size={17} /></button>
      <button type="button" className="canvas-control" onClick={() => void fitView({ padding: 0.25, duration: 500 })} aria-label="Fit graph"><Scan size={17} /></button>
    </Panel>
  );
}

function GraphWorkspaceInner({ trace, activeGraphId, citationFocus, summarize }: GraphWorkspaceProps) {
  const workspaceRef = useRef<HTMLDivElement>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<TraceNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedId, setSelectedId] = useState<string>();
  const [showNeighbours, setShowNeighbours] = useState(false);
  const pendingFocus = useRef<string>();
  const { fitView, getNode, setCenter } = useReactFlow();

  // Each new question opens on what it touched.
  useEffect(() => setShowNeighbours(false), [trace.id]);

  const shown = useMemo(() => questionGraph(trace.graph, showNeighbours), [trace.graph, showNeighbours]);
  const hiddenNeighbours = trace.graph.nodes.length - questionGraph(trace.graph, false).nodes.length;

  const focus = useCallback((nodeId: string) => {
    const node = getNode(nodeId);
    if (!node) return false;
    setSelectedId(node.id);
    void setCenter(node.position.x + NODE_W / 2, node.position.y + NODE_H / 2, { zoom: 1.25, duration: 650 });
    return true;
  }, [getNode, setCenter]);

  useEffect(() => {
    setSelectedId(undefined);
    setNodes(toFlowNodes(shown));
    setEdges(toFlowEdges(shown));
    const timer = window.setTimeout(() => {
      // A citation to a neighbour that was hidden: focus it once it is drawn.
      const waiting = pendingFocus.current;
      pendingFocus.current = undefined;
      if (!waiting || !focus(waiting)) void fitView({ padding: 0.25, duration: 700 });
    }, 120);
    return () => window.clearTimeout(timer);
  }, [fitView, focus, setEdges, setNodes, shown, trace.id]);

  useEffect(() => {
    setNodes((current) => current.map((node) => ({ ...node, selected: node.id === selectedId })));
  }, [selectedId, setNodes]);

  useEffect(() => {
    if (!citationFocus) return;
    if (focus(citationFocus.nodeId)) return;
    if (trace.graph.nodes.some((node) => node.id === citationFocus.nodeId)) {
      pendingFocus.current = citationFocus.nodeId;
      setShowNeighbours(true);
    }
  }, [citationFocus?.nonce]);

  const selectedNode = useMemo(() => nodes.find((node) => node.id === selectedId)?.data, [nodes, selectedId]);
  const maxInspectorHeight = useCallback(() => workspaceRef.current?.clientHeight ?? 500, []);
  const handleNodesChange: OnNodesChange = useCallback((changes) => onNodesChange(changes), [onNodesChange]);

  return (
    <div ref={workspaceRef} className="relative h-full min-h-0 w-full overflow-hidden">
      <ReactFlow
        className="graph-flow" nodes={nodes} edges={edges} nodeTypes={nodeTypes}
        onNodesChange={handleNodesChange} onEdgesChange={onEdgesChange}
        onNodeClick={(_, node) => setSelectedId(node.id)} onPaneClick={() => setSelectedId(undefined)}
        nodesDraggable nodesConnectable={false} elementsSelectable minZoom={0.2} maxZoom={2.2}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="rgb(var(--m-ink-muted) / .25)" />
        <CanvasControls neighbours={{ shown: showNeighbours, hidden: hiddenNeighbours, onToggle: () => setShowNeighbours((value) => !value) }} />
      </ReactFlow>
      {selectedNode ? <NodeInspector node={selectedNode} activeGraphId={activeGraphId} maxHeight={maxInspectorHeight} onClose={() => setSelectedId(undefined)} summarize={summarize} /> : null}
    </div>
  );
}

export function GraphWorkspace(props: GraphWorkspaceProps) {
  return <ReactFlowProvider><GraphWorkspaceInner {...props} /></ReactFlowProvider>;
}
