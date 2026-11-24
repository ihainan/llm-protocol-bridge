package ir

import (
	"path"
	"strings"
)

// fileMediaTypes maps common document extensions to media types. It is fixed
// (not the host's MIME table) so conversions are deterministic everywhere.
var fileMediaTypes = map[string]string{
	".pdf":  "application/pdf",
	".txt":  "text/plain",
	".md":   "text/markdown",
	".csv":  "text/csv",
	".tsv":  "text/tab-separated-values",
	".html": "text/html",
	".htm":  "text/html",
	".xml":  "application/xml",
	".json": "application/json",
	".doc":  "application/msword",
	".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
	".xls":  "application/vnd.ms-excel",
	".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
	".ppt":  "application/vnd.ms-powerpoint",
	".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
	".odt":  "application/vnd.oasis.opendocument.text",
	".rtf":  "application/rtf",
	".epub": "application/epub+zip",
	".zip":  "application/zip",
}

// MediaTypeFromFilename guesses a document's media type from its file name
// extension, or returns "" when the extension is not a known document type.
func MediaTypeFromFilename(name string) string {
	return fileMediaTypes[strings.ToLower(path.Ext(name))]
}

// IsTextMediaType reports whether a media type is plain text that can be sent
// as text content (text/*, JSON, XML).
func IsTextMediaType(mt string) bool {
	mt = strings.ToLower(strings.TrimSpace(mt))
	if i := strings.Index(mt, ";"); i >= 0 {
		mt = strings.TrimSpace(mt[:i])
	}
	return strings.HasPrefix(mt, "text/") || mt == "application/json" || mt == "application/xml"
}
