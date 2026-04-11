/*
 * Luduan launcher stub
 *
 * A minimal compiled executable so macOS associates the running process
 * with Luduan.app (and its icon) rather than with the Python runtime.
 *
 * Locates the bundled venv Python relative to this executable, sets
 * HF_HOME so the Whisper model cache is always on a local path, then
 * exec()s Python in-place (no subprocess — same PID, same bundle).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mach-o/dyld.h>   /* _NSGetExecutablePath */
#include <libgen.h>
#include <sys/stat.h>

int main(int argc, char *argv[])
{
    /* ---- locate this executable ---- */
    char self[4096];
    uint32_t size = (uint32_t)sizeof(self);
    if (_NSGetExecutablePath(self, &size) != 0) {
        fprintf(stderr, "Luduan: could not resolve executable path\n");
        return 1;
    }

    /* ---- build path to bundled Python ---- */
    /* self is  …/Luduan.app/Contents/MacOS/Luduan          */
    /* python is …/Luduan.app/Contents/Resources/venv/bin/python */
    char *macosDir  = dirname(self);          /* …/MacOS             */
    char contentsDir[4096];
    snprintf(contentsDir, sizeof(contentsDir), "%s/..", macosDir);

    char python[4096];
    snprintf(python, sizeof(python),
             "%s/Resources/venv/bin/python", contentsDir);

    struct stat st;
    if (stat(python, &st) != 0) {
        fprintf(stderr,
            "Luduan: Python not found at %s\n"
            "        Run 'make app && make install' to rebuild the bundle.\n",
            python);
        return 1;
    }

    /* ---- set HF_HOME so model cache lands on a local path ---- */
    const char *home = getenv("HOME");
    if (home) {
        char hf_home[4096];
        snprintf(hf_home, sizeof(hf_home), "%s/.config/luduan/models", home);
        /* mkdir -p equivalent (best-effort, Python will also do this) */
        char mkdirCmd[4096];
        snprintf(mkdirCmd, sizeof(mkdirCmd), "mkdir -p '%s'", hf_home);
        system(mkdirCmd);
        setenv("HF_HOME", hf_home, 0 /* don't overwrite if already set */);
    }

    /* ---- exec Python in-place (replaces this process) ---- */
    char *new_argv[] = { python, "-m", "luduan.main", NULL };
    execv(python, new_argv);

    /* execv only returns on error */
    perror("Luduan: execv failed");
    return 1;
}
