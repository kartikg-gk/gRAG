import {
  CircleDot, File, FileText, FolderGit2, GitCommitHorizontal, GitPullRequest, User,
} from "lucide-react";

import type { EntityType } from "@/types/trace";

interface EntityIconProps { type: EntityType; size?: number }

export function EntityIcon({ type, size = 19 }: EntityIconProps) {
  const props = { size, strokeWidth: 1.8, "aria-hidden": true as const };
  switch (type) {
    case "Person": return <User {...props} />;
    case "PR": return <GitPullRequest {...props} />;
    case "Repo": return <FolderGit2 {...props} />;
    case "File": return <File {...props} />;
    case "Document": return <FileText {...props} />;
    case "Ticket": return <CircleDot {...props} />;
    case "Commit": return <GitCommitHorizontal {...props} />;
    default: return <FileText {...props} />;
  }
}
