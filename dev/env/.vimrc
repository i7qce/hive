set number              " show line numbers
set wrap                " wrap lines
set encoding=utf-8      " set encoding to UTF-8 (default was "latin1")
syntax on




"Custom commands for todo lists, to emulate behavior of macOs Notes

autocmd BufRead,BufNewFile *.todo :set nonu

autocmd BufRead,BufNewFile *.todo :syn match todoHeading /^[A-Z].*/
autocmd BufRead,BufNewFile *.todo :hi todoHeading cterm=underline ctermfg=darkgreen

autocmd BufRead,BufNewFile *.todo :syn match todo_header /^===.*/
autocmd BufRead,BufNewFile *.todo :hi todo_header cterm=bold,underline ctermfg=lightblue

autocmd BufRead,BufNewFile *.todo :syn match todo_inprogress /^.*\[_.*/ 
autocmd BufRead,BufNewFile *.todo :hi todo_inprogress cterm=bold ctermfg=blue

autocmd BufRead,BufNewFile *.todo :syn match todo_completed /^.*\[✓.*/
autocmd BufRead,BufNewFile *.todo :hi todo_completed ctermfg=darkgray

autocmd BufRead,BufNewFile *.todo :syn match checkmark /✓/ containedin=todo_completed
autocmd BufRead,BufNewFile *.todo :hi checkmark ctermfg=lightgreen

autocmd BufRead,BufNewFile *.todo :syn match date /‖.*/ containedin=todo_completed
autocmd BufRead,BufNewFile *.todo :hi date ctermfg=darkyellow

autocmd BufRead,BufNewFile *.todo nnoremap <CR> <Esc>o[ ] 
autocmd BufRead,BufNewFile *.todo inoremap <CR> <Esc>o[ ] 

autocmd BufRead,BufNewFile *.todo nnoremap ll <Esc>o<Esc>

autocmd BufRead,BufNewFile *.todo nnoremap <tab> <Esc>0i    <Esc>A
autocmd BufRead,BufNewFile *.todo inoremap <tab> <Esc>0i    <Esc>A

autocmd BufRead,BufNewFile *.todo nnoremap <S-tab> <Esc>0xxxx<Esc>A
autocmd BufRead,BufNewFile *.todo inoremap <S-tab> <Esc>0xxxx<Esc>A

autocmd BufRead,BufNewFile *.todo nnoremap qq ^xxi[✓<Esc>A ‖ <Esc>:r!date<ENTER>i<BS><Esc>0
autocmd BufRead,BufNewFile *.todo nnoremap ww ^xxi[ <Esc> /‖ <ENTER> hhD <Esc>k

autocmd BufRead,BufNewFile *.todo nnoremap cc ^xxi[_<Esc>

autocmd BufRead,BufNewFile *.todo nnoremap jj <Esc>:g/\[✓.*/m$<ENTER>:0<Enter> 