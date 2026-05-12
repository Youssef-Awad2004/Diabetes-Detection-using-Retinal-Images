"use client"

import * as React from "react"
import { CheckCircle, AlertTriangle, AlertCircle, Info, Activity } from "lucide-react"
import { cn } from "@/lib/utils"
import type { ClassificationResult, ClassificationCategory } from "@/lib/api-config"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"

interface ClassificationResultDisplayProps {
  result: ClassificationResult | null
  isLoading?: boolean
}

const categoryConfig: Record<
  ClassificationCategory,
  {
    label: string
    description: string
    icon: React.ElementType
    colorClass: string
    bgClass: string
    progressClass: string
  }
> = {
  normal: {
    label: "Normal",
    description: "No signs of diabetic retinopathy detected",
    icon: CheckCircle,
    colorClass: "text-success",
    bgClass: "bg-success/10",
    progressClass: "bg-success",
  },
  mild: {
    label: "Mild NPDR",
    description: "Mild non-proliferative diabetic retinopathy",
    icon: Info,
    colorClass: "text-chart-2",
    bgClass: "bg-chart-2/10",
    progressClass: "bg-chart-2",
  },
  moderate: {
    label: "Moderate NPDR",
    description: "Moderate non-proliferative diabetic retinopathy",
    icon: AlertTriangle,
    colorClass: "text-warning",
    bgClass: "bg-warning/10",
    progressClass: "bg-warning",
  },
  severe: {
    label: "Severe NPDR",
    description: "Severe non-proliferative diabetic retinopathy",
    icon: AlertCircle,
    colorClass: "text-severe",
    bgClass: "bg-severe/10",
    progressClass: "bg-severe",
  },
  proliferative: {
    label: "Proliferative DR",
    description: "Proliferative diabetic retinopathy - requires immediate attention",
    icon: AlertCircle,
    colorClass: "text-destructive",
    bgClass: "bg-destructive/10",
    progressClass: "bg-destructive",
  },
}

export function ClassificationResultDisplay({
  result,
  isLoading = false,
}: ClassificationResultDisplayProps) {
  if (isLoading) {
    return (
      <Card className="animate-pulse">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-3">
            <div className="size-10 rounded-full bg-muted" />
            <div className="space-y-2">
              <div className="h-5 w-32 rounded bg-muted" />
              <div className="h-4 w-48 rounded bg-muted" />
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="h-4 w-full rounded bg-muted" />
          <div className="h-20 w-full rounded bg-muted" />
        </CardContent>
      </Card>
    )
  }

  if (!result) {
    return (
      <Card className="border-dashed">
        <CardContent className="flex flex-col items-center justify-center py-12 text-center">
          <div className="flex size-16 items-center justify-center rounded-full bg-muted">
            <Activity className="size-8 text-muted-foreground" />
          </div>
          <CardTitle className="mt-4 text-lg">No Analysis Results</CardTitle>
          <CardDescription className="mt-2 max-w-sm">
            Upload a retinal fundus image and click &quot;Analyze Image&quot; to receive a
            classification result.
          </CardDescription>
        </CardContent>
      </Card>
    )
  }

  const config = categoryConfig[result.category]
  const Icon = config.icon

  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-3">
        <div className="flex items-start gap-4">
          <div
            className={cn(
              "flex size-12 shrink-0 items-center justify-center rounded-full",
              config.bgClass
            )}
          >
            <Icon className={cn("size-6", config.colorClass)} />
          </div>
          <div className="min-w-0 flex-1">
            <CardTitle className={cn("text-xl", config.colorClass)}>
              {config.label}
            </CardTitle>
            <CardDescription className="mt-1">
              {config.description}
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Confidence Score */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium text-foreground">Confidence Score</span>
            <span className={cn("font-semibold", config.colorClass)}>
              {(result.confidence * 100).toFixed(1)}%
            </span>
          </div>
          <div className="relative h-3 w-full overflow-hidden rounded-full bg-muted">
            <div
              className={cn("h-full transition-all duration-500", config.progressClass)}
              style={{ width: `${result.confidence * 100}%` }}
            />
          </div>
        </div>

        {/* Details Section */}
        {result.details && (
          <div className="space-y-3 rounded-lg bg-muted/50 p-4">
            <h4 className="text-sm font-semibold text-foreground">Analysis Details</h4>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {result.details.description}
            </p>
            {result.details.recommendations && result.details.recommendations.length > 0 && (
              <div className="mt-4 space-y-2">
                <h5 className="text-sm font-medium text-foreground">Recommendations</h5>
                <ul className="space-y-1.5">
                  {result.details.recommendations.map((rec, index) => (
                    <li
                      key={index}
                      className="flex items-start gap-2 text-sm text-muted-foreground"
                    >
                      <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-primary" />
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Disclaimer */}
        <div className="flex items-start gap-2 rounded-lg border border-warning/30 bg-warning/5 p-3">
          <AlertTriangle className="size-4 shrink-0 text-warning mt-0.5" />
          <p className="text-xs text-muted-foreground leading-relaxed">
            This analysis is for screening purposes only and should not replace professional
            medical diagnosis. Please consult an ophthalmologist for clinical evaluation.
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
