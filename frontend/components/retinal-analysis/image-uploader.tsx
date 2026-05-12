"use client"

import * as React from "react"
import { Upload, X, ImageIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { API_CONFIG } from "@/lib/api-config"

interface ImageUploaderProps {
  onImageSelect: (file: File) => void
  onImageClear: () => void
  selectedImage: File | null
  previewUrl: string | null
  disabled?: boolean
}

export function ImageUploader({
  onImageSelect,
  onImageClear,
  selectedImage,
  previewUrl,
  disabled = false,
}: ImageUploaderProps) {
  const [isDragOver, setIsDragOver] = React.useState(false)
  const inputRef = React.useRef<HTMLInputElement>(null)

  const validateFile = (file: File): string | null => {
    if (!API_CONFIG.ALLOWED_FILE_TYPES.includes(file.type)) {
      return `Invalid file type. Allowed types: ${API_CONFIG.ALLOWED_EXTENSIONS.join(", ")}`
    }
    if (file.size > API_CONFIG.MAX_FILE_SIZE) {
      return `File too large. Maximum size: ${API_CONFIG.MAX_FILE_SIZE / (1024 * 1024)}MB`
    }
    return null
  }

  const handleFileSelect = (file: File) => {
    const error = validateFile(file)
    if (error) {
      alert(error)
      return
    }
    onImageSelect(file)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    if (disabled) return

    const file = e.dataTransfer.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    if (!disabled) {
      setIsDragOver(true)
    }
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleClick = () => {
    if (!disabled) {
      inputRef.current?.click()
    }
  }

  return (
    <div className="w-full">
      <input
        ref={inputRef}
        type="file"
        accept={API_CONFIG.ALLOWED_FILE_TYPES.join(",")}
        onChange={handleInputChange}
        className="hidden"
        disabled={disabled}
      />

      {!selectedImage ? (
        <div
          onClick={handleClick}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={cn(
            "relative flex flex-col items-center justify-center gap-4 rounded-xl border-2 border-dashed p-8 transition-all cursor-pointer min-h-[280px]",
            isDragOver
              ? "border-primary bg-primary/5"
              : "border-border hover:border-primary/50 hover:bg-muted/50",
            disabled && "opacity-50 cursor-not-allowed"
          )}
        >
          <div className="flex size-16 items-center justify-center rounded-full bg-primary/10">
            <Upload className="size-8 text-primary" />
          </div>
          <div className="text-center">
            <p className="text-lg font-medium text-foreground">
              Upload Retinal Fundus Image
            </p>
            <p className="mt-1 text-sm text-muted-foreground">
              Drag and drop or click to browse
            </p>
            <p className="mt-2 text-xs text-muted-foreground">
              Supported formats: JPG, PNG, TIFF (max 10MB)
            </p>
          </div>
        </div>
      ) : (
        <div className="relative rounded-xl border bg-card overflow-hidden">
          <div className="relative aspect-square max-h-[400px] w-full">
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Selected retinal fundus image"
                className="size-full object-contain bg-muted/30"
              />
            ) : (
              <div className="flex size-full items-center justify-center bg-muted/30">
                <ImageIcon className="size-16 text-muted-foreground" />
              </div>
            )}
          </div>
          <div className="flex items-center justify-between border-t bg-muted/30 px-4 py-3">
            <div className="flex items-center gap-3 min-w-0">
              <ImageIcon className="size-5 text-muted-foreground shrink-0" />
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-foreground">
                  {selectedImage.name}
                </p>
                <p className="text-xs text-muted-foreground">
                  {(selectedImage.size / 1024).toFixed(1)} KB
                </p>
              </div>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation()
                onImageClear()
              }}
              disabled={disabled}
              className={cn(
                "flex size-8 items-center justify-center rounded-full bg-destructive/10 text-destructive hover:bg-destructive/20 transition-colors",
                disabled && "opacity-50 cursor-not-allowed"
              )}
              aria-label="Remove image"
            >
              <X className="size-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
