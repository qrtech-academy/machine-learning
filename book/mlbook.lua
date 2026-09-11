-- mlbook.lua - the one piece of the design that TeX macros cannot do cleanly.
--
-- \code{...} typesets C++ exactly as it is written in the lectures. \detokenize gets most of the
-- way, but leaves two artifacts that C++ runs into constantly: it doubles every #, so #include
-- would print as ##include, and it puts a space after every control word, so "\n" would print as
-- "\n ". Both are undone here. A line break is also allowed after a scope operator or a comma, so
-- a name like ml::dense_layer::Interface can break at a :: rather than run into the margin, and
-- inside the long CamelCase names the test suites give their cases, so
-- BackpropagateUsesPreActivationDerivative can break between two of its words.

local catcode_other = -2

-- A name this long or longer may break before a capital that follows a small letter.
local long_name = 24

-- The positions after which a break is allowed in s, in increasing order.
local function breakpoints(s)
  local after = {}
  local start = 1
  while true do
    local i, j = s:find("::", start, true)
    if not i then break end
    after[j] = true
    start = j + 1
  end
  -- A comma followed by a space needs nothing, because the space is a breakpoint already.
  for k in s:gmatch("(),[^ ]") do after[k] = true end
  for first, name in s:gmatch("()([%w_]+)") do
    if #name >= long_name then
      for k in name:gmatch("%l()%u") do after[first + k - 2] = true end
    end
  end
  local list = {}
  for k in pairs(after) do list[#list + 1] = k end
  table.sort(list)
  return list
end

function mlbook_code(s)
  s = s:gsub("#+", "#")
  s = s:gsub("(\\%a+) ", "%1")
  -- \%, \{ and \} are how a literal percent sign or an unbalanced brace has to be written inside
  -- a TeX argument, \# is how a # has to be written in a heading or a caption, and \\ is a
  -- lone backslash.
  s = s:gsub("\\([%%{}#\\])", "%1")
  -- Each tex.sprint is read as a line of its own and TeX skips the spaces a line starts with, so
  -- no chunk may begin with one; no breakpoint is ever followed by a space, so none does.
  local start = 1
  for _, j in ipairs(breakpoints(s)) do
    tex.sprint(catcode_other, s:sub(start, j))
    -- Discouraged rather than free, so a line breaks at a space when it can.
    tex.sprint("\\penalty100 ")
    start = j + 1
  end
  tex.sprint(catcode_other, s:sub(start))
end
