#include <sys/ioctl.h>
#include <stdio.h>
#include <unistd.h>

void get_terminal_size(size_t &cols, size_t &rows) {
/*
struct winsize w;
ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
columns = w.ws_col;
rows = w.ws_row;
*/
#ifdef TIOCGSIZE
  struct ttysize ts;
  ioctl(STDIN_FILENO, TIOCGSIZE, &ts);
  cols = ts.ts_cols;
  rows = ts.ts_lines;
#elif defined(TIOCGWINSZ)
  struct winsize ts;
  ioctl(STDIN_FILENO, TIOCGWINSZ, &ts);
  cols = ts.ws_col;
  rows = ts.ws_row;
#endif /* TIOCGSIZE */
}

size_t get_terminal_width() {
  size_t c, r;
  get_terminal_size(c, r);
  return c;
}

size_t get_terminal_height() {
  size_t c, r;
  get_terminal_size(c, r);
  return r;
}
