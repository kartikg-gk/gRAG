import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Minus, Plus, Scan } from "lucide-react";
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

function toFlowNodes(trace: TraceState): FlowNode[] {
  const laidOut = layoutGraph(trace.graph.nodes, trace.graph.edges);
  return laidOut.nodes.map((node) => ({
    id: node.id, type: "entity", position: node.position, data: node,
    sourcePosition: Position.Right, targetPosition: Position.Left,
    style: { width: NODE_W, height: NODE_H },
  }));
}

function toFlowEdges(trace: TraceState): Edge[] {
  return trace.graph.edges.map((edge) => ({
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

function CanvasControls() {
  const { fitView, zoomIn, zoomOut } = useReactFlow();
  return (
    <Panel position="top-right" className="!m-4 flex gap-2">
      <button type="button" className="canvas-control" onClick={() => void zoomIn({ duration: 180 })} aria-label="Zoom in"><Plus size={17} /></button>
      <button type="button" className="canvas-control" onClick={() => void zoomOut({ duration: 180 })} aria-label="Zoom out"><Minus size={17} /></button>
      <button type="button" className="canvas-control" onClick={() => void fitView({ padding: 0.25, duration: 500 })} aria-label="Fit graph"><Scan size={17} /></button>
    </Panel>
  );
}

function GraphWorkspaceInner({ trace, activeGraphId, citationFocus }: GraphWorkspaceProps) {
  const workspaceRef = useRef<HTMLDivElement>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<TraceNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedId, setSelectedId] = useState<string>();
  const { fitView, getNode, setCenter } = useReactFlow();

  useEffect(() => {
    setSelectedId(undefined);
    setNodes(toFlowNodes(trace));
    setEdges(toFlowEdges(trace));
    const timer = window.setTimeout(() => void fitView({ padding: 0.25, duration: 700 }), 120);
    return () => window.clearTimeout(timer);
  }, [fitView, setEdges, setNodes, trace, trace.graph, trace.id]);

  useEffect(() => {
    setNodes((current) => current.map((node) => ({ ...node, selected: node.id === selectedId })));
  }, [selectedId, setNodes]);

  useEffect(() => {
    if (!citationFocus) return;
    const node = getNode(citationFocus.nodeId);
    if (!node) return;
    setSelectedId(node.id);
    void setCenter(node.position.x + NODE_W / 2, node.position.y + NODE_H / 2, { zoom: 1.25, duration: 650 });
  }, [citationFocus?.nonce, getNode, setCenter]);

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
        <CanvasControls />
      </ReactFlow>
      {selectedNode ? <NodeInspector node={selectedNode} activeGraphId={activeGraphId} maxHeight={maxInspectorHeight} onClose={() => setSelectedId(undefined)} /> : null}
    </div>
  );
}

export function GraphWorkspace(props: GraphWorkspaceProps) {
  return <ReactFlowProvider><GraphWorkspaceInner {...props} /></ReactFlowProvider>;
}
