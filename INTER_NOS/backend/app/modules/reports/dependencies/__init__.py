"""Dependencias del módulo 7.

Política de acceso (la misma de los módulos 3 y 5, que gestionan sesiones y evaluaciones):

- `AcademicoUser` (módulo 1): token `access` de un docente o administrador activo. Los
  estudiante con token `access` recibe 403; los tokens `student` y `student_results`, 401
  (`decode_access_token` solo acepta el tipo `access`); sin token, 401.
- Además, cada servicio exige ser el docente RESPONSABLE de la sesión o el AUTOR del caso
  (404 si no).   docentes conservan el alcance de sus propias sesiones; el administrador consulta el
  catálogo académico completo.
"""

from typing import Annotated

from fastapi import Depends

from app.modules.auth.dependencies.auth import DbSession
from app.modules.reports.services import ExportacionService, ReportesAcademicosService, ReportesSesionService
from app.modules.sessions.dependencies import RelojDep


def get_reportes_sesion_service(db: DbSession, reloj: RelojDep) -> ReportesSesionService:
    return ReportesSesionService(db, reloj)


def get_reportes_academicos_service(db: DbSession, reloj: RelojDep) -> ReportesAcademicosService:
    return ReportesAcademicosService(db, reloj)


def get_exportacion_service(db: DbSession, reloj: RelojDep) -> ExportacionService:
    return ExportacionService(db, reloj)


ReportesSesionDep = Annotated[ReportesSesionService, Depends(get_reportes_sesion_service)]
ReportesAcademicosDep = Annotated[ReportesAcademicosService, Depends(get_reportes_academicos_service)]
ExportacionDep = Annotated[ExportacionService, Depends(get_exportacion_service)]
