import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { MainViewComponent } from './components/main-view/main-view.component';
import { AppIntroductionComponent } from './components/app-introduction/app-introduction.component';

const routes: Routes = [
  { path: '', component: AppIntroductionComponent },
  { path: 'main-view', component: MainViewComponent },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}