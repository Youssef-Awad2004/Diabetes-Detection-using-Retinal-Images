"use client"

import * as React from "react"
import { Loader2, Scan, RotateCcw } from "lucide-react"
import { ImageUploader } from "./image-uploader"
import { ClassificationResultDisplay } from "./classification-result"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { API_CONFIG, type ClassificationResult, type APIResponse } from "@/lib/api-config"

type AnalysisState = "idle" | "uploading" | "analyzing" | "complete" | "error"

/**
 * Mock classification function for demonstration purposes.
 * Replace this with actual API call when integrating with backend.
 */
async function classifyImage(file: File): Promise<APIResponse> {
  // Simulate network delay
  await new Promise((resolve) => setTimeout(resolve, 2000))

  // For demonstration, return a random classification
  // In production, this would be an actual API call:
  //
  // const formData = new FormData()
  // formData.append("image", file)
  //
  // const response = await fetch(API_CONFIG.CLASSIFICATION_API_URL, {
  //   method: "POST",
  //   body: formData,
  // })
  //
  // if (!response.ok) {
  //   throw new Error("Classification failed")
  // }
  //
  // return response.json()

  const categories: Array<{
    category: ClassificationResult["category"]
    confidence: number
    details: ClassificationResult["details"]
  }> = [
    {
      category: "normal",
      confidence: 0.94,
      details: {
        description:
          "The retinal image shows healthy fundus characteristics with clear optic disc margins, normal vessel caliber, and no signs of microaneurysms, hemorrhages, or exudates.",
        recommendations: [
          "Continue regular annual eye examinations",
          "Maintain good glycemic control",
          "Monitor blood pressure regularly",
        ],
      },
    },
    {
      category: "mild",
      confidence: 0.87,
      details: {
        description:
          "Early signs of diabetic retinopathy detected, including presence of microaneurysms. No significant hemorrhages or exudates observed at this stage.",
        recommendations: [
          "Schedule follow-up examination in 6-12 months",
          "Optimize blood sugar control",
          "Consult with your primary care physician about diabetes management",
        ],
      },
    },
    {
      category: "moderate",
      confidence: 0.82,
      details: {
        description:
          "Moderate non-proliferative diabetic retinopathy identified. Multiple microaneurysms and some hemorrhages present in multiple quadrants.",
        recommendations: [
          "Schedule ophthalmology appointment within 3-6 months",
          "Intensify diabetes management",
          "Regular blood pressure monitoring recommended",
          "Consider lipid management evaluation",
        ],
      },
    },
    {
      category: "severe",
      confidence: 0.91,
      details: {
        description:
          "Severe non-proliferative diabetic retinopathy detected. Significant hemorrhaging, cotton wool spots, and venous beading observed across multiple quadrants.",
        recommendations: [
          "Urgent ophthalmology referral recommended",
          "Consider evaluation for laser treatment",
          "Intensive glycemic control required",
          "Regular monitoring every 2-3 months",
        ],
      },
    },
  ]

  const randomResult = categories[Math.floor(Math.random() * categories.length)]

  return {
    success: true,
    result: randomResult,
  }
}

export function AnalysisForm() {
  const [selectedImage, setSelectedImage] = React.useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = React.useState<string | null>(null)
  const [analysisState, setAnalysisState] = React.useState<AnalysisState>("idle")
  const [progress, setProgress] = React.useState(0)
  const [result, setResult] = React.useState<ClassificationResult | null>(null)
  const [error, setError] = React.useState<string | null>(null)

  const handleImageSelect = (file: File) => {
    setSelectedImage(file)
    setResult(null)
    setError(null)
    setAnalysisState("idle")

    // Create preview URL
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
  }

  const handleImageClear = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }
    setSelectedImage(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
    setAnalysisState("idle")
    setProgress(0)
  }

  const handleAnalyze = async () => {
    if (!selectedImage) return

    setAnalysisState("uploading")
    setProgress(0)
    setError(null)

    try {
      // Simulate upload progress
      const uploadInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 40) {
            clearInterval(uploadInterval)
            return 40
          }
          return prev + 8
        })
      }, 100)

      // Wait for upload simulation
      await new Promise((resolve) => setTimeout(resolve, 600))
      clearInterval(uploadInterval)
      setProgress(40)

      setAnalysisState("analyzing")

      // Simulate analysis progress
      const analyzeInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) {
            clearInterval(analyzeInterval)
            return 90
          }
          return prev + 5
        })
      }, 150)

      // Call the classification API (mock for now)
      const response = await classifyImage(selectedImage)

      clearInterval(analyzeInterval)
      setProgress(100)

      if (response.success && response.result) {
        setResult(response.result)
        setAnalysisState("complete")
      } else {
        throw new Error(response.error || "Classification failed")
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred")
      setAnalysisState("error")
      setProgress(0)
    }
  }

  const handleReset = () => {
    handleImageClear()
  }

  const isProcessing = analysisState === "uploading" || analysisState === "analyzing"

  return (
    <div className="grid gap-8 lg:grid-cols-2">
      {/* Left Column - Upload Section */}
      <div className="space-y-6">
        <div>
          <h2 className="text-lg font-semibold text-foreground">Step 1: Upload Image</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Select a retinal fundus image for diabetic retinopathy screening
          </p>
        </div>

        <ImageUploader
          onImageSelect={handleImageSelect}
          onImageClear={handleImageClear}
          selectedImage={selectedImage}
          previewUrl={previewUrl}
          disabled={isProcessing}
        />

        {/* Progress Indicator */}
        {isProcessing && (
          <div className="space-y-3 rounded-lg border bg-card p-4">
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium text-foreground">
                {analysisState === "uploading" ? "Uploading image..." : "Analyzing image..."}
              </span>
              <span className="text-muted-foreground">{progress}%</span>
            </div>
            <Progress value={progress} className="h-2" />
            <p className="text-xs text-muted-foreground">
              {analysisState === "uploading"
                ? "Preparing image for analysis"
                : "AI model processing retinal patterns"}
            </p>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4">
            <p className="text-sm text-destructive">{error}</p>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-3">
          <Button
            onClick={handleAnalyze}
            disabled={!selectedImage || isProcessing}
            size="lg"
            className="flex-1 sm:flex-none"
          >
            {isProcessing ? (
              <>
                <Loader2 className="animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Scan />
                Analyze Image
              </>
            )}
          </Button>

          {(result || error) && (
            <Button
              onClick={handleReset}
              variant="outline"
              size="lg"
              disabled={isProcessing}
            >
              <RotateCcw />
              New Analysis
            </Button>
          )}
        </div>
      </div>

      {/* Right Column - Results Section */}
      <div className="space-y-6">
        <div>
          <h2 className="text-lg font-semibold text-foreground">Step 2: View Results</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Classification results and recommendations will appear here
          </p>
        </div>

        <ClassificationResultDisplay result={result} isLoading={isProcessing} />
      </div>
    </div>
  )
}
