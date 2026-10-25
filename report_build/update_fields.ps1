# Opens the report in Microsoft Word, updates the table of contents, the lists
# of figures/tables and all page numbers, saves it, and exports a PDF copy.
# Usage: powershell -File report_build\update_fields.ps1 <report.docx> <report.pdf>
param([string]$Docx, [string]$Pdf)
$Docx = (Resolve-Path $Docx).Path
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx)
    # Two passes: page numbers can shift once the lists are filled in.
    for ($i = 0; $i -lt 2; $i++) {
        foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
        $doc.Fields.Update() | Out-Null
    }
    $doc.Save()
    $doc.ExportAsFixedFormat($Pdf, 17)   # 17 = wdExportFormatPDF
    "pages: " + $doc.ComputeStatistics(2)  # 2 = wdStatisticPages
    $doc.Close()
} finally {
    $word.Quit()
}
