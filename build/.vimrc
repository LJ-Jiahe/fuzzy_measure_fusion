""""""""""""""""""""""""""""""""""
" # Initialize: Basic Needs
""""""""""""""""""""""""""""""""""

syntax on 

set ruler
set number 
set nowrap
set statusline+=%F

""""""""""""""""""""""""""""""""""
" # Initialize: Python Essential
""""""""""""""""""""""""""""""""""

set tabstop=8
set expandtab
set softtabstop=4
set shiftwidth=4

""""""""""""""""""""""""""""""""""
" # Initialize: Custom Keybinds
""""""""""""""""""""""""""""""""""

map = :tabn<CR>
map - :tabp<CR>
inoremap <C-E> <Esc>
vnoremap <C-E> <Esc>

""""""""""""""""""""""""""""""""""
" # Initialize: Color Theme
""""""""""""""""""""""""""""""""""

colo elflord

hi TabLineSel ctermfg=Red ctermbg=Black
hi TabLine ctermfg=White ctermbg=Black
hi TabLineFill ctermfg=Black ctermbg=Black
